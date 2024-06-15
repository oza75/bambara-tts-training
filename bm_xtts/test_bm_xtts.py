from bm_features import XTTSFeatureExtractor, XTTSProcessor, XTTSTokenizer
from datasets import load_dataset
from bm_models import XttsConfig, Xtts
import datasets
import torch

DEVICE = torch.device('mps')
dataset = load_dataset("oza75/bambara-tts", "denoised")
dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=22050)).rename_column('bambara', 'text')

feature_extractor = XTTSFeatureExtractor("../mel_stats.pth", sampling_rate=22050, max_samples=221000)
tokenizer = XTTSTokenizer.from_pretrained("openai-community/gpt2-medium")
processor = XTTSProcessor(feature_extractor, tokenizer)

batch = processor(dataset['train'][:4], device=DEVICE)

model = Xtts.from_pretrained("./bm_xtts")
model.to(DEVICE)

if __name__ == "__main__":
    model(
        input_ids=batch['text_tokens'],
        label_ids=batch['text_tokens'],  # same as text_tokens
        text_attn_masks=batch['text_attn_masks'],
        cond_16k=batch['cond_16k'],
        cond_mels=batch['cond_mels'],
        orig_wavs=batch['orig_wavs'],
        wav_mels=batch['wav_mels'],
        wav_lengths=batch['wav_lengths'],
    )
