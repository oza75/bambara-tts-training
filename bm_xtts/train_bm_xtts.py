import datasets
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, EvalPrediction, TrainerCallback
from transformers.integrations import WandbCallback
import wandb
import numpy as np
from matplotlib import pyplot as plt

from bm_features import XTTSFeatureExtractor, XTTSTokenizer, XTTSProcessor
from bm_models import Xtts, XttsConfig

DEVICE = torch.device("mps")


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
        speaker_embeddings = batch["speaker_embeddings"].to(model.device)

        # Get model outputs
        with torch.no_grad():
            loss, outputs = model(inputs, attention_masks, speaker_embeddings=speaker_embeddings)

        # Apply attention masks to original and reconstructed mel spectrograms
        inputs_masked = inputs * attention_masks
        outputs_masked = outputs * attention_masks

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
    logits, wavs, vq_loss, loss_text_ce, loss_mel_ce, wav_loss = eval_pred.predictions

    return {
        "vq_loss": vq_loss.item(),
        "loss_text_ce": loss_text_ce.item(),
        "loss_mel_ce": loss_mel_ce.item(),
        "wav_loss": wav_loss.item()
    }


class DataCollator:
    """
    Custom data collator to handle input values and attention masks.
    """

    def __call__(self, features):
        batch = {}
        # keys = [
        #     'wav_mels', 'cond_16k', 'cond_len', 'cond_idxs', 'cond_mels', 'wav',
        #     'orig_wavs', 'wav_lengths', 'text_tokens', 'text_attn_masks', 'text_lengths', 'input_ids', 'label_ids',
        #     'cond_lens'
        # ]
        for feature in features:
            for k in feature.keys():
                batch[k] = batch[k] if k in batch else []
                batch[k].append(torch.tensor(feature[k], device=DEVICE))

        for k in batch.keys():
            batch[k] = torch.stack(batch[k]).to(device=DEVICE) if k not in ['orig_wavs'] else batch[k]

        return batch


def preprocess_function(examples, processor):
    """
    Preprocess the dataset examples using the processor.

    Args:
        examples (dict): The input examples from the dataset.
        processor (VQVAEProcessor): The processor for extracting features.

    Returns:
        dict: The preprocessed examples with input values and attention masks.
    """
    batch = processor(examples, device=DEVICE)

    return dict(
        input_ids=batch['text_tokens'],
        label_ids=batch['text_tokens'],  # same as text_tokens
        text_lengths=batch['text_lengths'],
        text_attn_masks=batch['text_attn_masks'],
        cond_16k=batch['cond_16k'],
        cond_mels=batch['cond_mels'],
        cond_idxs=batch['cond_idxs'],
        cond_lens=batch['cond_len'],
        orig_wavs=batch['orig_wavs'],
        wav_mels=batch['wav_mels'],
        wav_lengths=batch['wav_lengths'],
    )


def main():
    # Load the Bambara TTS dataset
    dataset = load_dataset("oza75/bambara-tts", "denoised")
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=22050)).rename_column('bambara', 'text')

    # Split the dataset into training and evaluation sets
    dataset = dataset['train'].select(range(50)).train_test_split(test_size=0.1, seed=42)

    # Instantiate the feature extractor and processor
    feature_extractor = XTTSFeatureExtractor("../mel_stats.pth", sampling_rate=22050, max_samples=221000)
    tokenizer = XTTSTokenizer.from_pretrained("openai-community/gpt2-medium")
    processor = XTTSProcessor(feature_extractor, tokenizer)

    # Preprocess the datasets
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, processor),
        batched=True,
        batch_size=1000,
        num_proc=1,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    checkpoint_path = None

    if checkpoint_path is not None:
        model = Xtts.from_pretrained(checkpoint_path)
    else:
        config = XttsConfig(
            gpt_number_text_tokens=processor.tokenizer.vocab_size,
            gpt_start_text_token=processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.START_TOKEN),
            gpt_stop_text_token=processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.STOP_TOKEN),
        )
        # Define the model
        model = Xtts(config)

    model.to(DEVICE)
    # Define training arguments
    training_args = TrainingArguments(
        # torch_compile=True,
        output_dir="./results",
        eval_strategy="steps",
        logging_steps=1,
        eval_steps=1,
        save_steps=200,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        warmup_steps=500,
        optim="adamw_hf",
        remove_unused_columns=True,
        # dataloader_drop_last=True,
        # dataloader_num_workers=16,
        # dataloader_prefetch_factor=4,
        # dataloader_persistent_workers=True,
        # dataloader_pin_memory=True,
        # ddp_find_unused_parameters=False,
        # fp16=True,
        # deepspeed="../deepspeed_config.json",
        # report_to=['tensorboard', 'wandb'],
        # debug="underflow_overflow"
        report_to=['tensorboard'],
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
        # callbacks=[AudioLoggingCallback]
    )

    # torch.autograd.set_detect_anomaly(True)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
