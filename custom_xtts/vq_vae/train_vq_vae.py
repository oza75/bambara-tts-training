import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments, Trainer, EvalPrediction
import evaluate
from .features import VQVAEProcessor, VQVAEFeatureExtractor
from .models import SpeechVQVAE

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

def compute_metrics(eval_pred: EvalPrediction):
    """
    Compute evaluation metrics for VQ-VAE.

    Args:
        eval_pred (EvalPrediction): The predictions and labels.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    logits, labels = eval_pred
    mse = ((logits - labels) ** 2).mean().item()
    return {"mse": mse}

class DataCollator:
    """
    Custom data collator to handle input values and attention masks.
    """
    def __call__(self, features):
        input_values = torch.stack([f["input_values"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        return {"input_values": input_values, "attention_mask": attention_mask}

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

def main():
    # Load the Bambara TTS dataset
    dataset = load_dataset("oza75/bambara-tts", "denoised")

    # Split the dataset into training and evaluation sets
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)

    # Instantiate the feature extractor and processor
    feature_extractor = VQVAEFeatureExtractor(sampling_rate=22050, mel_norm_file='../mel_stats.pth', max_duration=10)
    processor = VQVAEProcessor(feature_extractor)

    # Preprocess the datasets
    dataset = dataset.map(lambda examples: preprocess_function(examples, processor), batched=True)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Define the VQ-VAE model
    model = SpeechVQVAE(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=1,
        act_fn="silu",
        latent_channels=1,
        sample_size=32,
        num_vq_embeddings=256,
        norm_num_groups=32,
        vq_embed_dim=None,
        scaling_factor=0.18215,
        norm_type="group"
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        deepspeed="./deepspeed_config.json",
        logging_dir="./logs",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
