import torchaudio
import numpy as np
from transformers import FeatureExtractionMixin, ProcessorMixin
import torch.nn as nn
import torch


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


class VQVAEFeatureExtractor(FeatureExtractionMixin):
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
        self.max_samples = max_samples

    def __call__(self, audio_batch):
        """
        Process a batch of audio examples.

        Args:
            audio_batch (list of dict): A batch of dictionaries, each containing a raw audio waveform.

        Returns:
            dict: A dictionary containing the batch of normalized Mel spectrograms and attention masks.
        """
        waveforms = []
        attention_masks = []
        for audio in audio_batch:
            # Extract the audio array from the input dictionary
            waveform = audio["array"]

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

        # Return a dictionary containing the Mel spectrograms and attention masks
        return {"mel_spectrogram": mel_spectrograms, "attention_masks": attention_masks}


class VQVAEProcessor(ProcessorMixin):
    """
    Processor for handling the feature extraction and preprocessing pipeline.

    Attributes:
        feature_extractor (VQVAEFeatureExtractor): The feature extractor instance.
    """
    attributes = []

    def __init__(self, feature_extractor):
        """
        Initialize the processor.

        Args:
            feature_extractor (VQVAEFeatureExtractor): The feature extractor instance.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.attributes = ["feature_extractor"]

    def __call__(self, batch):
        """
        Process a batch of audio data.

        Args:
            batch (dict): A batch containing raw audio waveforms.

        Returns:
            dict: A batch containing Mel spectrograms.
        """
        features = self.feature_extractor(batch['audio'])
        batch["mel_spectrogram"] = features["mel_spectrogram"]
        batch["attention_masks"] = features["attention_masks"]
        return batch
