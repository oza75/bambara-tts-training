import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, EvalPrediction, TrainerCallback
from transformers.integrations import WandbCallback
from features import VQVAEProcessor, VQVAEFeatureExtractor
from models import SpeechVQVAE, SpeechVQConfig
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
        inputs = batch["input_values"].to(model.device)
        attention_masks = batch["attention_mask"].to(model.device)

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


def masked_vq_vae_loss(x, x_recon, z_e, z_q, attention_mask, beta=0.25):
    """
    Compute the masked VQ-VAE loss.

    Args:
        x (torch.Tensor): Original input.
        x_recon (torch.Tensor): Reconstructed input.
        z_e (torch.Tensor): Encoder output (continuous).
        z_q (torch.Tensor): Quantized encoder output (discrete).
        attention_mask (torch.Tensor): Mask indicating valid parts of the input.
        beta (float): Hyperparameter for commitment loss.

    Returns:
        torch.Tensor: Total VQ-VAE loss with masking.
    """
    # Ensure the attention mask has the same shape as the inputs
    attention_mask = attention_mask.expand_as(x)

    # Reconstruction loss with masking
    recon_loss = F.mse_loss(x_recon * attention_mask, x * attention_mask, reduction='none')
    recon_loss = recon_loss.sum() / attention_mask.sum()

    # VQ loss with masking
    vq_loss = F.mse_loss(z_q.detach() * attention_mask, z_e * attention_mask, reduction='none')
    vq_loss = vq_loss.sum() / attention_mask.sum()

    # Commitment loss with masking
    commit_loss = F.mse_loss(z_q * attention_mask, z_e.detach() * attention_mask, reduction='none')
    commit_loss = commit_loss.sum() / attention_mask.sum()

    # Total loss
    loss = recon_loss + vq_loss + beta * commit_loss

    return loss


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for VQ-VAE, applying attention masks.

    Args:
        eval_pred (EvalPrediction): The predictions and labels.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    (logits, attention_masks), labels = eval_pred.predictions, eval_pred.label_ids

    print(f"Logits type: {type(logits)}, Labels type: {type(labels)}, Attention masks type: {type(attention_masks)}")
    print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}, Attention masks shape: {attention_masks.shape}")

    # Ensure logits, labels, and attention_masks are tensors
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    if isinstance(attention_masks, np.ndarray):
        attention_masks = torch.tensor(attention_masks)

    print(f"Compute Metrics: {labels.shape, attention_masks.shape}")
    # Apply the attention masks
    logits = logits * attention_masks
    labels = labels * attention_masks

    # Compute masked Mean Squared Error (MSE)
    mse = F.mse_loss(logits, labels, reduction='none')
    masked_mse = mse.sum() / attention_masks.sum()

    # Compute Signal-to-Noise Ratio (SNR)
    signal_power = (labels ** 2).sum() / attention_masks.sum()
    noise_power = ((labels - logits) ** 2).sum() / attention_masks.sum()
    snr = 10 * torch.log10(signal_power / noise_power)

    return {"mse": masked_mse.item(), "snr": snr.item()}


class DataCollator:
    """
    Custom data collator to handle input values and attention masks.
    """

    def __call__(self, features):
        input_values = torch.stack([torch.tensor(f["input_values"], dtype=torch.float32) for f in features])
        attention_mask = torch.stack([torch.tensor(f["attention_mask"], dtype=torch.float32) for f in features])

        batch_size, n_mels, time_steps = input_values.shape

        input_values = input_values.view(batch_size, 1, n_mels, time_steps)
        attention_mask = attention_mask.view(batch_size, 1, n_mels, time_steps)

        return {"input_values": input_values, "labels": input_values, "attention_mask": attention_mask}


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
        "input_values": batch["mel_spectrogram"],
        "attention_mask": batch["attention_mask"]
    }


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_values = inputs["input_values"]
        attention_mask = inputs["attention_mask"]

        # Forward pass
        outputs = model(input_values, attention_mask=attention_mask)
        logits = outputs[0]

        # Compute custom loss
        loss = masked_vq_vae_loss(
            x=input_values,
            x_recon=logits,
            z_e=outputs[2],
            z_q=outputs[1],
            attention_mask=attention_mask,
            beta=0.25
        )

        return (loss, outputs) if return_outputs else loss


def main():
    # Load the Bambara TTS dataset
    dataset = load_dataset("oza75/bambara-tts", "denoised")

    # Split the dataset into training and evaluation sets
    dataset = dataset['train'].select(range(20)).train_test_split(test_size=0.1, seed=42)

    # Instantiate the feature extractor and processor
    feature_extractor = VQVAEFeatureExtractor(sampling_rate=22050, mel_norm_file='../mel_stats.pth', max_duration=10)
    processor = VQVAEProcessor(feature_extractor)

    # Preprocess the datasets
    dataset = dataset.map(lambda examples: preprocess_function(examples, processor), batched=True, num_proc=8)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    config = SpeechVQConfig()

    # Define the VQ-VAE model
    model = SpeechVQVAE(config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        eval_steps=1,
        save_steps=1000,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        tf32=True,
        deepspeed="../deepspeed_config.json",
        report_to=['tensorboard', 'wandb'],
    )

    data_collator = DataCollator()

    # Initialize the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[AudioLoggingCallback]
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
