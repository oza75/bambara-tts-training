import datasets
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainingArguments, Trainer, EvalPrediction, TrainerCallback
from transformers.integrations import WandbCallback
import wandb
import numpy as np
from matplotlib import pyplot as plt

from bm_features import XTTSFeatureExtractor, XTTSTokenizer, XTTSProcessor
from bm_models import Xtts, XttsConfig

DEVICE = torch.device("mps")

# Instantiate the feature extractor and processor
feature_extractor = XTTSFeatureExtractor("../mel_stats.pth", sampling_rate=22050, max_samples=221000)
tokenizer = XTTSTokenizer.from_pretrained("openai-community/gpt2-medium")
processor = XTTSProcessor(feature_extractor, tokenizer)


class AudioLoggingCallback(WandbCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        # Ensure the trainer and model are available
        model = kwargs.get("model")
        tokenizer: XTTSTokenizer = kwargs.get("tokenizer")
        eval_dataloader = kwargs.get("eval_dataloader")

        if eval_dataloader is None or model is None:
            print(f"Either eval_dataloader or model aren't provided: model {model}, eval_dataloader {eval_dataloader}")
            return

        # Get a batch from the validation dataset
        batch = next(iter(eval_dataloader))

        # Get model outputs
        with torch.no_grad():
            outputs = model(**batch)

        for idx, wav in enumerate(outputs['wavs']):
            text = tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=True)
            self._wandb.log({
                f"audio_{idx}": wandb.Audio(wav.cpu().numpy(), sample_rate=22050, caption=text)
            })


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

        for feature in features:
            for k in feature.keys():
                batch[k] = batch[k] if k in batch else []
                batch[k].append(torch.tensor(feature[k], device=DEVICE))

        for k in batch.keys():
            if k == 'text_attn_masks':
                batch[k] = pad_sequence(batch[k], batch_first=True, padding_value=0)
            elif k in ['input_ids', 'label_ids']:
                batch[k] = pad_sequence(batch[k], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
            else:
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
    dataset = dataset['train'].select(range(50)).filter(lambda ex: [x < 221000 / 22050 for x in ex['duration']],
                                                        batched=True, batch_size=50)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Preprocess the datasets
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, processor),
        batched=True,
        batch_size=20,
        num_proc=1,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    checkpoint_path = "./bm_xtts"

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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=50,
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
        report_to=['tensorboard', 'wandb'],
        # debug="underflow_overflow"
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
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[AudioLoggingCallback]
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
