from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from transformers import PretrainedConfig

from bm_xtts.bm_utils.gpt import GPT
from bm_xtts.bm_utils.hifigan_decoder import HifiDecoder
from bm_xtts.bm_utils.vq_vae import BMSpeechVQVAE


class XttsConfig(PretrainedConfig):
    model_type = "xtts"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # General config parameters
        self.gpt_batch_size = kwargs.get('gpt_batch_size', 1)
        self.enable_redaction = kwargs.get('enable_redaction', False)
        self.kv_cache = kwargs.get('kv_cache', True)
        self.gpt_checkpoint = kwargs.get('gpt_checkpoint')
        self.clvp_checkpoint = kwargs.get('clvp_checkpoint')
        self.decoder_checkpoint = kwargs.get('decoder_checkpoint')
        self.num_chars = kwargs.get('num_chars', 255)

        # XTTS GPT Encoder parameters
        self.gpt_max_audio_tokens = kwargs.get('gpt_max_audio_tokens', 605)
        self.gpt_max_text_tokens = kwargs.get('gpt_max_text_tokens', 402)
        self.gpt_max_prompt_tokens = kwargs.get('gpt_max_prompt_tokens', 70)
        self.gpt_layers = kwargs.get('gpt_layers', 30)
        self.gpt_n_model_channels = kwargs.get('gpt_n_model_channels', 1024)
        self.gpt_n_heads = kwargs.get('gpt_n_heads', 16)
        self.gpt_number_text_tokens = kwargs.get('gpt_number_text_tokens')
        self.gpt_start_text_token = kwargs.get('gpt_start_text_token')
        self.gpt_stop_text_token = kwargs.get('gpt_stop_text_token')
        self.gpt_num_audio_tokens = kwargs.get('gpt_num_audio_tokens', 1026)
        self.gpt_start_audio_token = kwargs.get('gpt_start_audio_token', 1024)
        self.gpt_stop_audio_token = kwargs.get('gpt_stop_audio_token', 1025)
        self.gpt_code_stride_len = kwargs.get('gpt_code_stride_len', 1024)
        self.gpt_use_masking_gt_prompt_approach = kwargs.get('gpt_use_masking_gt_prompt_approach', True)
        self.gpt_use_perceiver_resampler = kwargs.get('gpt_use_perceiver_resampler', True)

        # HifiGAN Decoder parameters
        self.input_sample_rate = kwargs.get('input_sample_rate', 22050)
        self.output_sample_rate = kwargs.get('output_sample_rate', 24000)
        self.output_hop_length = kwargs.get('output_hop_length', 256)
        self.decoder_input_dim = kwargs.get('decoder_input_dim', 1024)
        self.d_vector_dim = kwargs.get('d_vector_dim', 512)
        self.cond_d_vector_in_each_upsampling_layer = kwargs.get('cond_d_vector_in_each_upsampling_layer', True)

        # Constants
        self.duration_const = kwargs.get('duration_const', 102400)

        # Additional parameters
        # self.min_conditioning_length = kwargs.get('min_conditioning_length', 66150)
        # self.max_conditioning_length = kwargs.get('max_conditioning_length', 132300)
        # self.max_wav_length = kwargs.get('max_wav_length', 255995)  # ~11.6 seconds
        self.gpt_loss_text_ce_weight = kwargs.get('gpt_loss_text_ce_weight', 0.01)
        self.gpt_loss_mel_ce_weight = kwargs.get('gpt_loss_mel_ce_weight', 1.0)
        self.debug_loading_failures = kwargs.get('debug_loading_failures', False)
        self.max_text_length = kwargs.get('max_text_length', 200)
        # self.mel_norm_file = kwargs.get('mel_norm_file', "https://coqui.gateway.scarf.sh/v0.14.0_models/mel_norms.pth")
        self.dvae_checkpoint = kwargs.get('dvae_checkpoint', "")
        self.vqvae_checkpoint = kwargs.get('vqvae_checkpoint', "oza75/bambara-vqvae")
        self.xtts_checkpoint = kwargs.get('xtts_checkpoint', "")
        self.vocoder = kwargs.get('vocoder', "")


class Xtts(PreTrainedModel):
    config_class = XttsConfig

    def __init__(self, config: XttsConfig):
        super().__init__(config)

        self.gpt = GPT(
            layers=config.gpt_layers,
            model_dim=config.gpt_n_model_channels,
            heads=config.gpt_n_heads,
            start_text_token=config.gpt_start_text_token,
            stop_text_token=config.gpt_stop_text_token,
            max_text_tokens=config.gpt_max_text_tokens,
            max_mel_tokens=config.gpt_max_audio_tokens,
            max_prompt_tokens=config.gpt_max_prompt_tokens,
            number_text_tokens=config.gpt_number_text_tokens,
            num_audio_tokens=config.gpt_num_audio_tokens,
            start_audio_token=config.gpt_start_audio_token,
            stop_audio_token=config.gpt_stop_audio_token,
            use_perceiver_resampler=config.gpt_use_perceiver_resampler,
            code_stride_len=config.gpt_code_stride_len,
        )

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=config.input_sample_rate,
            output_sample_rate=config.output_sample_rate,
            output_hop_length=config.output_hop_length,
            ar_mel_length_compression=config.gpt_code_stride_len,
            decoder_input_dim=config.decoder_input_dim,
            d_vector_dim=config.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=config.cond_d_vector_in_each_upsampling_layer,
        )

        self.vq_vae = BMSpeechVQVAE.from_pretrained(config.vqvae_checkpoint)

    def inference_forward(self):
        self.gpt.init_gpt_for_inference()
        self.gpt.eval()

    def forward(
            self,
            input_ids: torch.LongTensor,  # text_tokens
            label_ids: torch.LongTensor,  # same as text_tokens
            text_lengths: torch.LongTensor,
            cond_mels: torch.FloatTensor = None,
            cond_idxs: torch.LongTensor = None,
            cond_lens: torch.LongTensor = None,
            wav_mels: torch.FloatTensor = None,
            wav_mel_attention_masks: torch.FloatTensor = None,
            wav_lengths: torch.LongTensor = None,
    ) -> dict:
        audio_codes, vq_loss = self.vq_vae.get_codebook_indices(wav_mels)

        loss_text, loss_mel, mel_logits = self.gpt(
            input_ids,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
        )

        outputs = {
            "logits": mel_logits,
            "vq_loss": vq_loss,
            "loss_text_ce": loss_text * self.config.gpt_loss_text_ce_weight,
            "loss_mel_ce": loss_mel * self.config.gpt_loss_mel_ce_weight
        }

        outputs["loss"] = outputs["loss_text_ce"] + outputs["loss_mel_ce"]

        return outputs
