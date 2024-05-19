from typing import Optional, Union, Tuple, Any

import torch
from diffusers import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.vq_model import VQEncoderOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from transformers import PretrainedConfig, PreTrainedModel


class VQVAE(VQModel):
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
            self,
            input_ids: torch.FloatTensor,
            attention_masks: Optional[torch.Tensor] = None,
            return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor, ...]]:
        """
        The forward method with attention mask support.

        Args:
            input_ids (`torch.FloatTensor`): Input values.
            attention_masks (`torch.Tensor`, *optional*): Attention mask for the input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `DecoderOutput` instead of a plain tuple.

        Returns:
            `DecoderOutput` or `tuple`:
                If return_dict is True, a `DecoderOutput` is returned, otherwise a plain `tuple` is returned.
        """
        encoder_outputs = self.encode(input_ids, return_dict=return_dict)

        if not return_dict:
            h_e, h_q = encoder_outputs
        else:
            h_q = encoder_outputs.latents

        dec = self.decode(h_q).sample

        if not return_dict:
            return dec, h_q, h_e

        return DecoderOutput(sample=dec)


class SpeechVQConfig(PretrainedConfig):
    model_type = "speech_vq_vae"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = kwargs.get("in_channels", 1)
        self.out_channels = kwargs.get("out_channels", 1)
        self.down_block_types = kwargs.get("down_block_types", ("DownEncoderBlock2D",))
        self.up_block_types = kwargs.get("up_block_types", ("UpDecoderBlock2D",))
        self.block_out_channels = kwargs.get("block_out_channels", (64,))
        self.layers_per_block = kwargs.get("layers_per_block", 1)
        self.act_fn = kwargs.get("act_fn", "silu")
        self.latent_channels = kwargs.get("latent_channels", 1)
        self.sample_size = kwargs.get("sample_size", 32)
        self.num_vq_embeddings = kwargs.get("num_vq_embeddings", 256)
        self.norm_num_groups = kwargs.get("norm_num_groups", 32)
        self.vq_embed_dim = kwargs.get("vq_embed_dim", None)
        self.scaling_factor = kwargs.get("scaling_factor", 0.18215)
        self.norm_type = kwargs.get("norm_type", "group")


class SpeechVQVAE(PreTrainedModel):
    config_class = SpeechVQConfig

    def __init__(self, config: SpeechVQConfig, model: Optional[VQVAE] = None):
        super().__init__(config)
        self.model = model if model is not None else VQVAE(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            latent_channels=config.latent_channels,
            sample_size=config.sample_size,
            num_vq_embeddings=config.num_vq_embeddings,
            norm_num_groups=config.norm_num_groups,
            vq_embed_dim=config.vq_embed_dim,
            scaling_factor=config.scaling_factor,
            norm_type=config.norm_type
        )

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> Union[Tuple[Any, Any], VQEncoderOutput]:
        return self.model.encode(x, return_dict=return_dict)

    def forward(
            self,
            input_ids: torch.FloatTensor,
            label_ids: torch.FloatTensor = None,
            attention_masks: Optional[torch.Tensor] = None,
            return_dict: bool = False
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor, ...]]:
        return self.model(input_ids, attention_masks=attention_masks, return_dict=return_dict)

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        self.model.save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = SpeechVQConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = VQVAE.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return cls(config, model)
