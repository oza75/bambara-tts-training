import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, EvalPrediction, TrainerCallback
from transformers.integrations import WandbCallback
from features import VQVAEProcessor, VQVAEFeatureExtractor
from models import BMSpeechVQVAE, BMSpeechVQVAEConfig
import wandb
import numpy as np
from matplotlib import pyplot as plt


class AudioLoggingCallback(WandbCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        # Ensure the trainer and model are available
        model = kwargs.get("model")
        eval_dataloader = kwargs.get("eval_dataloader")

        if eval_dataloader is None or model is None:
            print(f"Either eval_dataloader or model aren't provided: model {model}, eval_dataloader {eval_dataloader}")
            return

        # Get a batch from the validation dataset
        batch = next(iter(eval_dataloader))

        # Move batch to the correct device
        inputs = batch["input_ids"].to(model.device)
        attention_masks = batch["attention_masks"].to(model.device)

        # Get model outputs
        with torch.no_grad():
            outputs = model(inputs, return_dict=True)

        # Apply attention masks to original and reconstructed mel spectrograms
        inputs_masked = inputs * attention_masks
        outputs_masked = outputs.sample * attention_masks

        # Log original and reconstructed mel spectrograms
        for idx in range(min(5, inputs.size(0))):  # Log up to 5 samples
            original_mel = inputs_masked[idx].cpu().squeeze(0).numpy()
            reconstructed_mel = outputs_masked[idx].squeeze(0).cpu().numpy()

            # Extract valid sections based on the attention mask
            valid_length = int(attention_masks[idx].squeeze(0)[0].sum().item())
            original_mel_valid = original_mel[:, :valid_length]
            reconstructed_mel_valid = reconstructed_mel[:, :valid_length]

            # Plot and save the mel spectrograms
            fig, axs = plt.subplots(2, 1, figsize=(10, 6))
            axs[0].imshow(original_mel_valid, aspect='auto', origin='lower')
            axs[0].set_title('Original Mel Spectrogram')
            axs[1].imshow(reconstructed_mel_valid, aspect='auto', origin='lower')
            axs[1].set_title('Reconstructed Mel Spectrogram')
            plt.tight_layout()

            # Log using wandb
            self._wandb.log({
                f"mel_spectrogram_{idx}": wandb.Image(fig)
            })

            # Close the plot to free memory
            plt.close(fig)


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for VQ-VAE.

    Args:
        eval_pred (EvalPrediction): The predictions and labels.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # Ensure logits and labels are tensors
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    # Ensure the shapes are the same
    if logits.shape != labels.shape:
        print(f"Shape mismatch: logits shape {logits.shape} != labels shape {labels.shape}")
        return {"mse": float('inf'), "snr": float('-inf')}  # Return default values if shapes don't match

    # Compute Mean Squared Error (MSE)
    mse = F.mse_loss(logits, labels, reduction='mean')

    # Compute Signal-to-Noise Ratio (SNR)
    signal_power = (labels ** 2).mean()
    noise_power = ((labels - logits) ** 2).mean()
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-6))  # Add epsilon for numerical stability

    return {"mse": mse.item(), "snr": snr.item()}


class DataCollator:
    """
    Custom data collator to handle input values and attention masks.
    """

    def __call__(self, features):
        input_ids = torch.stack([torch.tensor(f["input_ids"], dtype=torch.float32) for f in features])
        attention_masks = torch.stack([torch.tensor(f["attention_masks"], dtype=torch.float32) for f in features])

        batch_size, n_mels, time_steps = input_ids.shape

        input_ids = input_ids.view(batch_size, 1, n_mels, time_steps)
        attention_masks = attention_masks.view(batch_size, 1, n_mels, time_steps)

        return {"input_ids": input_ids, "label_ids": input_ids, "attention_masks": attention_masks}


def preprocess_function(examples, processor):
    """
    Preprocess the dataset examples using the processor.

    Args:
        examples (dict): The input examples from the dataset.
        processor (VQVAEProcessor): The processor for extracting features.

    Returns:
        dict: The preprocessed examples with input values and attention masks.
    """
    batch = processor(examples)

    return {
        "input_ids": batch["mel_spectrogram"],
        "attention_masks": batch["attention_masks"]
    }


def main():
    # Load the Bambara TTS dataset
    dataset = load_dataset("oza75/bambara-tts", "denoised")

    # Split the dataset into training and evaluation sets
    dataset = dataset['train'].select(range(250)).train_test_split(test_size=0.1, seed=42)

    # Instantiate the feature extractor and processor
    feature_extractor = VQVAEFeatureExtractor(sampling_rate=22050, mel_norm_file='../mel_stats.pth', max_samples=221000)
    processor = VQVAEProcessor(feature_extractor)

    # Preprocess the datasets
    dataset = dataset.map(lambda examples: preprocess_function(examples, processor), batched=True, batch_size=10, num_proc=2)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    config = BMSpeechVQVAEConfig()

    # Define the VQ-VAE model
    model = BMSpeechVQVAE(config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        logging_steps=3,
        eval_steps=3,
        save_steps=3,
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=500,
        optim="adamw_torch_fused",
        tf32=True,
        deepspeed="../deepspeed_config.json",
        report_to=['tensorboard', 'wandb'],
        # report_to=['tensorboard'],
    )

    data_collator = DataCollator()

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[AudioLoggingCallback]
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
