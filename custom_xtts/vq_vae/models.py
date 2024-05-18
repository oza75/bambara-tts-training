from typing import Optional, Union, Tuple, Any

import torch
from diffusers import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.vq_model import VQEncoderOutput
from diffusers.utils.accelerate_utils import apply_forward_hook


class SpeechVQVAE(VQModel):
    """
    A VQ-VAE model with support for attention masks for decoding latent representations.

    This class inherits from VQModel and adds support for attention masks.
    """

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> tuple[Any, Any] | VQEncoderOutput:
        h_e = self.encoder(x)
        h_q = self.quant_conv(h_e)

        if not return_dict:
            return h_e, h_q

        return VQEncoderOutput(latents=h_q)

    def forward(
            self, sample: torch.FloatTensor, attention_mask: Optional[torch.Tensor] = None, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor, ...]]:
        """
        The forward method with attention mask support.

        Args:
            sample (`torch.FloatTensor`): Input sample.
            attention_mask (`torch.Tensor`, *optional*): Attention mask for the input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `DecoderOutput` instead of a plain tuple.

        Returns:
            `DecoderOutput` or `tuple`:
                If return_dict is True, a `DecoderOutput` is returned, otherwise a plain `tuple` is returned.
        """
        h_e, h_q = self.encode(sample, return_dict=return_dict)

        dec = self.decode(h_q).sample

        if not return_dict:
            return dec, h_q, h_e

        return DecoderOutput(sample=dec)
