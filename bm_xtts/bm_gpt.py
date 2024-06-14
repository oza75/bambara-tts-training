import functools
import random

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

from bm_utils.perceiver_encoder import PerceiverResampler
from bm_utils.latent_encoder import ConditioningEncoder


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = torch.nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class GPTWrapper(nn.Module):
    def __init__(
            self,
            start_text_token=261,
            stop_text_token=0,
            layers=8,
            model_dim=512,
            heads=8,
            max_text_tokens=120,
            max_mel_tokens=250,
            max_prompt_tokens=70,
            max_conditioning_inputs=1,
            code_stride_len=1024,
            number_text_tokens=256,
            num_audio_tokens=8194,
            start_audio_token=8192,
            stop_audio_token=8193,
            train_solo_embeddings=False,
            checkpointing=False,
            average_conditioning_embeddings=False,
            label_smoothing=0.0,
            use_perceiver_resampler=False,
            perceiver_cond_length_compression=256,
    ):
        super().__init__()

        self.perceiver_cond_length_compression = perceiver_cond_length_compression
        self.label_smoothing = label_smoothing
        self.start_audio_token = start_audio_token
        self.stop_audio_token = stop_audio_token
        self.code_stride_len = code_stride_len
        self.text_emb = nn.Embedding(number_text_tokens, model_dim)
        self.mel_emb = nn.Embedding(max_mel_tokens, model_dim)

        self.gpt, self.text_pos_emb, self.mel_pos_emb = self.build_gpt_model(
            layers=layers,
            heads=heads,
            model_dim=model_dim,
            max_text_tokens=max_text_tokens,
            max_mel_tokens=max_mel_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, number_text_tokens)
        self.mel_head = nn.Linear(model_dim, num_audio_tokens)

        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
        self.conditioning_dropout = nn.Dropout1d(0.1)
        self.conditioning_perceiver = PerceiverResampler(
            dim=model_dim,
            depth=2,
            dim_context=model_dim,
            num_latents=32,
            dim_head=64,
            heads=8,
            ff_mult=4,
            use_flash_attn=False,
        )

    def build_gpt_model(
            self,
            layers: int,
            heads: int,
            model_dim: int,
            max_text_tokens: int,
            max_mel_tokens: int,
            max_prompt_tokens: int
    ):
        max_seq_length = max_text_tokens + max_mel_tokens + max_prompt_tokens
        config = GPT2Config(
            n_embd=model_dim,
            n_head=heads,
            n_layer=layers,
            n_positions=max_seq_length,
            n_ctx=max_seq_length,
        )

        gpt = GPT2Model(config)

        # Override the built in positional embeddings
        del gpt.wpe
        gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)

        # Built-in token embeddings are unused.
        del gpt.wte

        text_pos_emb = LearnedPositionEmbeddings(max_text_tokens, model_dim)
        mel_pos_emb = LearnedPositionEmbeddings(max_mel_tokens, model_dim)

        return gpt, text_pos_emb, mel_pos_emb

    def forward(
            self,
            text_inputs,
            text_lengths,
            text_attn_masks,
            audio_codes,
            wav_lengths,
            cond_mels=None,
            cond_idxs=None,
            cond_lens=None,
            cond_latents=None,
            return_attentions=False,
            return_latent=False,
    ):
        audio_codes, audio_attn_masks = self.format_audio_codes(audio_codes, wav_lengths)

        # Compute text embeddings + positional embeddings
        text_emb = self.text_emb(text_inputs) + self.text_pos_emb(text_inputs)

        # Compute mel embeddings + positional embeddings
        mel_emb = self.mel_emb(audio_codes) + self.mel_pos_emb(audio_codes)

        # Compute speech conditioning input
        if cond_latents is None:
            cond_latents = self.get_style_emb(cond_mels)
            cond_latents = cond_latents.transpose(1, 2)

        text_logits, mel_logits, text_latents, mel_latents = self.get_logits(
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            prompt=cond_latents,
            get_attns=return_attentions,
            return_latent=return_latent,
            attn_mask_text=text_attn_masks,
            attn_mask_mel=audio_attn_masks,
        )

        if return_latent or return_attentions:
            return mel_logits, mel_latents, audio_attn_masks

        # Set paddings to -1 to ignore them in loss
        text_targets = torch.where(text_attn_masks == 1, text_inputs, -100)
        mel_targets = torch.where(audio_attn_masks, audio_codes, -100)

        # Compute losses
        loss_text = F.cross_entropy(
            text_logits, text_targets.long(), ignore_index=-100, label_smoothing=self.label_smoothing
        )
        loss_mel = F.cross_entropy(
            mel_logits, mel_targets.long(), ignore_index=-100, label_smoothing=self.label_smoothing
        )

        return loss_text.mean(), loss_mel.mean(), mel_logits, mel_latents, audio_attn_masks

    def get_logits(
            self,
            text_emb,
            text_head,
            mel_emb=None,
            mel_head=None,
            prompt=None,
            get_attns=False,
            return_latent=False,
            attn_mask_text=None,
            attn_mask_mel=None,
    ):
        if prompt is not None:
            offset = prompt.shape[1]
            if mel_emb is not None:
                emb = torch.cat([prompt, text_emb, mel_emb], dim=1)
            else:
                emb = torch.cat([prompt, text_emb], dim=1)

        attn_mask = None
        if attn_mask_text is not None:
            attn_mask = torch.cat([attn_mask_text, attn_mask_mel], dim=1)
            if prompt is not None:
                attn_mask_cond = torch.ones(prompt.shape[0], offset, dtype=torch.bool, device=emb.device)
                attn_mask = torch.cat([attn_mask_cond, attn_mask], dim=1)

        gpt_out = self.gpt(
            inputs_embeds=emb,
            return_dict=True,
            output_attentions=get_attns,
            attention_mask=attn_mask,
        )

        if get_attns:
            return gpt_out.attentions

        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        text_latents, mel_latents = enc[:, : text_emb.shape[1]], enc[:, -mel_emb.shape[1]:]

        if return_latent:
            # return enc[:, : text_emb.shape[1]], enc[:, -mel_emb.shape[1]:]
            return text_latents, mel_latents

        # text_logits = enc[:, : text_emb.shape[1]]
        text_logits = text_head(text_latents)
        text_logits = text_logits.permute(0, 2, 1)
        if mel_emb is not None:
            # mel_logits = enc[:, -mel_emb.shape[1]:]
            mel_logits = mel_head(mel_latents)
            mel_logits = mel_logits.permute(0, 2, 1)
            return text_logits, mel_logits, text_latents, mel_latents
        else:
            return text_logits, text_latents

    def format_audio_codes(self, audio_codes, wav_lengths):
        """
        Reformats audio token sequences based on actual audio lengths by padding and truncating to fit the required
        length. This function ensures that all audio sequences are of the same length and terminated properly with
        a stop token.

        Args:
            audio_codes (torch.Tensor): Tensor of audio token indices with shape (batch_size, sequence_length).
            wav_lengths (torch.Tensor): Tensor containing the actual lengths of each wav file in the batch.

        Returns:
            torch.Tensor: The adjusted tensor of audio token indices with stop tokens appended and unnecessary
                          tokens removed based on the actual audio lengths.
        """
        # Compute the required number of tokens for each audio segment
        code_lengths = torch.ceil(wav_lengths / self.code_stride_len).long()
        max_mel_len = code_lengths.max()

        audio_attn_masks = torch.ones(
            audio_codes.shape[0],
            audio_codes.shape[1],
            dtype=torch.bool,
            device=audio_codes.device,
        )

        # Ensure all audio sequences are of the same maximum length
        if max_mel_len > audio_codes.shape[1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[1]))

        # Replace padding with stop_audio_token for each batch element
        audio_codes_padded = audio_codes.clone()
        for i in range(len(code_lengths)):
            actual_end = code_lengths[i]
            if actual_end < audio_codes.shape[1]:
                audio_codes_padded[i, actual_end:] = self.stop_audio_token

            audio_attn_masks[i, actual_end + 3:] = 0.0  # attn_mask for the audio_codes

        # Append a stop token at the end and start token at the beginning of each sequence
        audio_codes_padded = F.pad(audio_codes_padded[:, :max_mel_len], (0, 1), value=self.stop_audio_token)
        audio_codes_padded = F.pad(audio_codes_padded, (1, 0), value=self.start_audio_token)

        audio_attn_masks = audio_attn_masks[:, :max_mel_len + 2]  # take only the valid masks according to max_length

        return audio_codes_padded, audio_attn_masks

    def get_style_emb(self, cond_input, return_latent=False):
        """
        Processes conditioning input to extract style embeddings using a series of neural network layers.
        This function supports either generating embeddings through the network or directly using provided latent embeddings.

        Args:
            cond_input (torch.Tensor): The conditioning input which can either be a tensor with dimensions
                                       (batch, 80, sequence_length) or (batch, 1, 80, sequence_length), depending
                                       on whether the input is already expanded.
            return_latent (bool): Flag to determine whether to return the provided `cond_input` as latent embeddings
                                  after possibly adjusting dimensions, or to compute new embeddings using the model's
                                  layers. If True, the input is assumed to be pre-computed embeddings.

        Returns:
            torch.Tensor: The style embeddings as a tensor with dimensions adjusted to the model's requirements.
                          It provides a context for TTS generation based on the input features.
        """
        if return_latent:
            # If returning pre-computed latent embeddings, adjust dimensions to match expected format
            conds = cond_input.unsqueeze(1)
        else:
            # Process the conditioning input to generate new style embeddings
            if cond_input.ndim == 4:
                # Remove the singleton dimension if present (batch, 1, 80, sequence_length)
                cond_input = cond_input.squeeze(1)

            # Apply the conditioning encoder to the input to obtain initial embeddings
            conds = self.conditioning_encoder(cond_input)  # (batch, dimension, sequence_length)

            # Apply dropout to the encoder outputs for regularization
            conds = self.conditioning_dropout(conds)

            # The conditioning perceiver further processes the embeddings, potentially reducing dimensionality
            # and focusing on salient features for the TTS task
            # (batch, dimension, new_sequence_length)
            conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)

        return conds
