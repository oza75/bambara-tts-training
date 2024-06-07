import torchaudio
import numpy as np
from tokenizers import AddedToken
from transformers import FeatureExtractionMixin, ProcessorMixin, BertTokenizerFast, WhisperTokenizer, \
    WhisperTokenizerFast, GPT2Tokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F
import re


class TorchMelSpectrogram(nn.Module):
    """
    Mel spectrogram extractor for audio processing.

    This class converts raw audio waveforms to Mel spectrograms and normalizes them
    using precomputed normalization statistics.

    Attributes:
        filter_length (int): Length of the FFT window.
        hop_length (int): Number of audio samples between adjacent STFT columns.
        win_length (int): Window size.
        n_mel_channels (int): Number of Mel filter banks.
        mel_fmin (float): Minimum frequency for the Mel filter banks.
        mel_fmax (float): Maximum frequency for the Mel filter banks.
        sampling_rate (int): Audio sample rate.
        mel_stft (torchaudio.transforms.MelSpectrogram): Torchaudio Mel spectrogram transform.
        mel_norms (torch.Tensor): Normalization factors for the Mel spectrograms.
    """

    def __init__(
            self,
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=80,
            mel_fmin=0,
            mel_fmax=8000,
            sampling_rate=22050,
            normalize=False,
            mel_norm_file=None,
    ):
        """
        Initialize the Mel spectrogram extractor.

        Args:
            filter_length (int): Length of the FFT window.
            hop_length (int): Number of audio samples between adjacent STFT columns.
            win_length (int): Window size.
            n_mel_channels (int): Number of Mel filter banks.
            mel_fmin (float): Minimum frequency for the Mel filter banks.
            mel_fmax (float): Maximum frequency for the Mel filter banks.
            sampling_rate (int): Audio sample rate.
            normalize (bool): Whether to normalize the Mel spectrogram.
            mel_norm_file (str): Path to the file containing Mel normalization statistics.
        """
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=2,
            normalized=normalize,
            sample_rate=self.sampling_rate,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=self.n_mel_channels,
            norm="slaney",
        )
        self.mel_norms = None
        if mel_norm_file is not None:
            self.load_mel_norms(mel_norm_file)

    def load_mel_norms(self, mel_norm_file):
        """
        Load Mel normalization statistics from a file.

        Args:
            mel_norm_file (str): Path to the file containing Mel normalization statistics.
        """
        self.mel_norms = torch.load(mel_norm_file)

    def forward(self, inp):
        """
        Convert waveform to normalized Mel spectrogram.

        Args:
            inp (torch.Tensor): Raw audio waveform.

        Returns:
            torch.Tensor: Normalized Mel spectrogram.
        """
        # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
        if len(inp.shape) == 3:
            inp = inp.squeeze(1)

        assert len(inp.shape) == 2, "Input tensor must have shape (batch_size, num_samples)"

        # Move the Mel spectrogram transform to the same device as the input
        self.mel_stft = self.mel_stft.to(inp.device)

        # Compute the Mel spectrogram
        mel = self.mel_stft(inp)

        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # Normalize the Mel spectrogram if normalization factors are provided
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)

        return mel


class XTTSFeatureExtractor(FeatureExtractionMixin):
    """
    Feature extractor for converting raw audio waveforms to Mel spectrograms.

    Attributes:
        sampling_rate (int): The sample rate of the audio data.
        max_samples (int): The maximum max samples.
        n_mels (int): The number of Mel bands to generate.
    """

    def __init__(self, mel_norm_file=None, sampling_rate=16000, max_samples=160000):
        """
        Initialize the feature extractor.

        Args:
            sampling_rate (int): The sample rate of the audio data.
            mel_norm_file (str, optional): Path to the file containing Mel normalization statistics.
            max_samples (int, optional): Maximum duration (in seconds) * sample_rate of audio to process. Default is 10 seconds * 16000 sample_rate = 160000.
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.mel_spectrogram_extractor = TorchMelSpectrogram(
            mel_norm_file=mel_norm_file,
            sampling_rate=sampling_rate
        )

        self.mel_spectrogram_style_encoder = TorchMelSpectrogram(
            filter_length=4096,
            hop_length=1024,
            win_length=4096,
            normalize=False,
            sampling_rate=sampling_rate,
            mel_fmin=0,
            mel_fmax=8000,
            n_mel_channels=80,
            mel_norm_file=mel_norm_file,
        )

        self.max_samples = max_samples
        self.max_conditioning_length = 132300  # 6 secs
        self.min_conditioning_length = 66150  # 3 secs

    def __call__(self, audio_batch, is_eval=False):
        """
        Process a batch of audio examples.

        Args:
            audio_batch (list of dict): A batch of dictionaries, each containing a raw audio waveform.

        Returns:
            dict: A dictionary containing the batch of normalized Mel spectrograms and attention masks.
        """
        waveforms = []
        waveform_lengths = []
        attention_masks = []
        for audio in audio_batch:
            # Extract the audio array from the input dictionary
            waveform = audio["array"]
            waveform_lengths.append(torch.tensor(len(waveform), dtype=torch.long))

            # Convert the audio array to a tensor if it is not already a tensor
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)

            # Truncate or pad the waveform to the maximum length
            if waveform.size(0) > self.max_samples:
                waveform = waveform[:self.max_samples]
                attention_mask = torch.ones(self.max_samples, dtype=torch.long)
            else:
                padding = self.max_samples - waveform.size(0)
                attention_mask = torch.cat([
                    torch.ones(waveform.size(0), dtype=torch.long),
                    torch.zeros(padding, dtype=torch.long)
                ])
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            waveforms.append(waveform)
            attention_masks.append(attention_mask)

        # Stack the waveforms and attention masks into tensors
        waveforms = torch.stack(waveforms)
        # Extract Mel spectrograms
        mel_spectrograms = self.mel_spectrogram_extractor(waveforms)

        # Calculate the new size for the attention masks
        batch_size, n_mels, n_frames = mel_spectrograms.shape

        # Generate the attention masks at the time step level
        for idx, waveform in enumerate(waveforms):
            attention_mask = attention_masks[idx].float()
            attention_mask = torch.nn.functional.interpolate(
                attention_mask.unsqueeze(0).unsqueeze(0),
                size=n_frames,
                mode='nearest'
            ).squeeze()
            attention_masks[idx] = attention_mask

        # Stack the attention masks and reshape to match the mel spectrograms
        attention_masks = torch.stack(attention_masks).view(batch_size, 1, n_frames).expand(-1, n_mels, -1)
        cond, cond_len, cond_idxs = self.get_prompt_slice(waveforms, is_eval=is_eval)
        cond_mels = self.get_conditioning_mel_spectrogram(cond)

        # Return a dictionary containing the Mel spectrograms and attention masks
        return {
            "mel_spectrogram": mel_spectrograms,
            "attention_masks": attention_masks,
            "waveforms": waveforms,
            "waveform_lengths": torch.stack(waveform_lengths),
            "cond": cond,
            "cond_len": cond_len,
            "cond_idxs": cond_idxs,
            "cond_mels": cond_mels,
        }

    def get_prompt_slice(self, audio_batch, is_eval=False):
        if is_eval:
            # Use a constant sample length for all evaluation mode samples
            sample_lengths = [int((self.min_conditioning_length + self.max_conditioning_length) / 2)]
        else:
            # Generate random lengths for each sample if not in evaluation mode
            sample_lengths = np.random.randint(
                self.min_conditioning_length,
                self.max_conditioning_length,
                size=len(audio_batch)
            )

        conditioning = []
        conditioning_idxs = []
        conditioning_len = []

        for idx, sample in enumerate(audio_batch):
            rel_clip = sample.unsqueeze(0)
            if is_eval:
                # Use a fixed sample length for eval mode
                effective_sample_length = sample_lengths[0]
            else:
                # Use randomly generated lengths for each sample in training mode
                effective_sample_length = sample_lengths[idx]

            gap = max(rel_clip.shape[-1] - effective_sample_length, 0)

            if is_eval:
                rand_start = 0
            else:
                rand_start = np.random.randint(0, gap + 1) if gap > 0 else 0

            rand_end = rand_start + effective_sample_length
            rel_clip = rel_clip[:, rand_start:rand_end]
            rel_clip = F.pad(rel_clip, pad=(0, self.max_conditioning_length - rel_clip.shape[-1]))
            cond_idxs = [rand_start, rand_end]
            conditioning.append(rel_clip)
            conditioning_idxs.append(torch.tensor(cond_idxs))
            # conditioning_len.append(torch.tensor(rel_clip.shape[-1]))
            conditioning_len.append(torch.tensor(torch.nan))

        return torch.stack(conditioning), torch.tensor(conditioning_len), torch.stack(conditioning_idxs)

    def get_conditioning_mel_spectrogram(self, conditioning: torch.Tensor):
        # compute conditioning mel specs
        # transform waves from torch.Size([B, num_cond_samples, 1, T] to torch.Size([B * num_cond_samples, 1, T] because if is faster than iterate the tensor
        paired_conditioning_mel = self.mel_spectrogram_style_encoder(conditioning)
        return paired_conditioning_mel


class XTTSTokenizer(GPT2Tokenizer):
    # Define the special tokens
    START_TOKEN = "<|start|>"
    STOP_TOKEN = "<|stop|>"
    SPACE_TOKEN = "<|space|>"

    # Supported languages dictionary
    SUPPORTED_LANGUAGES = {
        'bm': 'Bambara',
    }

    def __init__(
            self,
            vocab_file=None,
            errors="replace",
            unk_token="<|unk|>",
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            add_prefix_space=False,
            add_bos_token=False,
            **kwargs
    ):
        super().__init__(
            vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            errors=errors,
            **kwargs,
        )

        additional_tokens = [self.START_TOKEN, self.STOP_TOKEN, self.SPACE_TOKEN] + [
            f"<|{lang}|>" for lang in self.SUPPORTED_LANGUAGES
        ]

        self.add_tokens(additional_tokens, special_tokens=False)

    def encode_plus(self, text, lang='bm', **kwargs):
        """
        Encodes the input text, prepending the language token and adding start and stop tokens.

        Args:
            text (str): The input text to encode.
            lang (str): The language code for the text. Default is 'bm' (Bambara).
            kwargs: Additional arguments for the parent class's encode_plus method.

        Returns:
            dict: A dictionary with encoded inputs and additional information.
        """
        assert lang in self.SUPPORTED_LANGUAGES, f"Unsupported language code: {lang}"

        # Prepend the language token
        lang_token = f"<|{lang}|>"
        text = f"{lang_token}{text}"

        # Encode the text using the parent class method
        return super().encode_plus(text, **kwargs)

    def batch_encode_plus(self, batch_text_or_text_pairs, lang='bm', **kwargs):
        """
        Encodes the input text, prepending the language token and adding start and stop tokens.

        Args:
            batch_text_or_text_pairs (List[str]): The input texts to encode.
            lang (str): The language code for the text. Default is 'bm' (Bambara).
            kwargs: Additional arguments for the parent class's encode_plus method.

        Returns:
            dict: A dictionary with encoded inputs and additional information.
        """
        assert lang in self.SUPPORTED_LANGUAGES, f"Unsupported language code: {lang}"

        # Prepend the language token
        lang_token = f"<|{lang}|>"
        texts = [f"{lang_token}{text}" for text in batch_text_or_text_pairs]

        # Encode the text using the parent class method
        return super().batch_encode_plus(texts, **kwargs)

    def decode(self, token_ids, **kwargs):
        """
        Decodes the token ids to the original text, removing special tokens.

        Args:
            token_ids (list[int]): List of token ids to decode.
            kwargs: Additional arguments for the parent class's decode method.

        Returns:
            str: The decoded text.
        """
        # Decode the tokens using the parent class method
        text = super().decode(token_ids, **kwargs)

        # # Remove special tokens
        # text = text.replace(self.start_token, "").replace(self.stop_token, "").replace(self.space_token,
        #                                                                                " ").strip()
        #
        # for lang in self.SUPPORTED_LANGUAGES:
        #     text = text.replace(f"[{lang}]", "").strip()

        return text


class XTTSProcessor(ProcessorMixin):
    """
    Processor for handling the feature extraction and preprocessing pipeline.

    Attributes:
        feature_extractor (XTTSFeatureExtractor): The feature extractor instance.
    """
    attributes = []

    def __init__(self, feature_extractor, tokenizer):
        """
        Initialize the processor.

        Args:
            feature_extractor (XTTSFeatureExtractor): The feature extractor instance.
        """
        super().__init__()
        self.feature_extractor: XTTSFeatureExtractor = feature_extractor
        self.tokenizer: XTTSTokenizer = tokenizer
        self.attributes = ["feature_extractor", "tokenizer"]

    def __call__(self, batch):
        """
        Process a batch of audio data.

        Args:
            batch (dict): A batch containing raw audio waveforms.

        Returns:
            dict: A batch containing Mel spectrograms.
        """
        features = self.feature_extractor(batch['audio'])
        text_features = self.tokenizer(batch['text'], lang="bm", padding=True, return_tensors="pt")
        batch["wav_mels"] = features["mel_spectrogram"]
        batch["wav_mel_attention_masks"] = features["attention_masks"]
        batch["cond"] = features["cond"]
        batch["cond_len"] = features["cond_len"]
        batch["cond_idxs"] = features["cond_idxs"]
        batch["cond_mels"] = features["cond_mels"]
        batch["wav"] = features["waveforms"]
        batch["wav_lengths"] = features["waveform_lengths"]
        batch["speaker_embeddings"] = torch.stack([torch.tensor(x) for x in batch['speaker_embeddings']])
        batch["text_tokens"] = text_features["input_ids"]
        batch["text_attention_masks"] = text_features["attention_mask"]
        batch["text_lengths"] = torch.sum(batch["text_attention_masks"], dim=1)

        return batch
