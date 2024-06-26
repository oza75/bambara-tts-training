from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class Encoder(nn.Module):
    def __init__(
            self,
            input_channels=1,
            num_layers=6,
            initial_filters=64,
            kernel_size=4,
            stride=2,
            padding=1,
            act_fn: nn.Module = nn.ReLU(),
            norm_num_groups=32,
    ):
        super(Encoder, self).__init__()
        layers = []
        in_channels = input_channels

        for i in range(num_layers):
            out_channels = initial_filters * (2 ** i)
            if in_channels % norm_num_groups == 0:
                layers.append(
                    nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels)
                )

            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            layers.append(act_fn)
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        self.latent_dim = out_channels

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(
            self,
            output_channels=1,
            output_shape=None,
            num_layers=6,
            initial_filters=64,
            kernel_size=4,
            stride=2,
            padding=1,
            act_fn: nn.Module = nn.ReLU(),
            norm_num_groups=32,
    ):
        super(Decoder, self).__init__()
        layers = []
        in_channels = initial_filters * (2 ** (num_layers - 1))

        for i in range(num_layers):
            out_channels = initial_filters * (2 ** (num_layers - i - 1)) // 2
            if i == num_layers - 1:
                out_channels = output_channels

            if in_channels % norm_num_groups == 0:
                layers.append(
                    nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels)
                )

            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            layers.append(act_fn)
            in_channels = out_channels

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
            self,
            n_e: int,
            vq_embed_dim: int,
            beta: float,
            remap=None,
            unknown_index: str = "random",
            sane_index_shape: bool = False,
            legacy: bool = False,
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.used: torch.Tensor
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, Tuple]:
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q: torch.FloatTensor = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Tuple[int, ...]) -> torch.FloatTensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q: torch.FloatTensor = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class BMSpeechVQVAEConfig(PretrainedConfig):
    model_type = "bm_speech_vq_vae"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = kwargs.get("input_shape", None)
        self.in_channels = kwargs.get("in_channels", 1)
        self.out_channels = kwargs.get("out_channels", 1)
        self.latent_channels = kwargs.get("latent_channels", 512)
        self.num_layers = kwargs.get("num_layers", 4)
        self.initial_filters = kwargs.get("initial_filters", 64)
        self.kernel_size = kwargs.get("kernel_size", 4)
        self.stride = kwargs.get("stride", 2)
        self.padding = kwargs.get("padding", 1)
        self.act_fn = kwargs.get("act_fn", "silu")
        self.sample_size = kwargs.get("sample_size", 32)
        self.num_vq_embeddings = kwargs.get("num_vq_embeddings", 512)
        self.vq_embed_dim = kwargs.get("vq_embed_dim", 64)
        self.speaker_embed_dim = kwargs.get("speaker_embed_dim", 512)
        self.scaling_factor = kwargs.get("scaling_factor", 0.18215)
        self.norm_type = kwargs.get("norm_type", "group")
        self.norm_num_groups = kwargs.get("norm_num_groups", 32)


class BMSpeechVQVAE(PreTrainedModel):
    config_class = BMSpeechVQVAEConfig
    ACTIVATIONS = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "silu": torch.nn.SiLU}

    def __init__(self, config: BMSpeechVQVAEConfig):
        super().__init__(config)
        act_fn = self.ACTIVATIONS[str(config.act_fn).lower()]
        self.encoder = Encoder(
            input_channels=config.in_channels,
            num_layers=config.num_layers,
            initial_filters=config.initial_filters,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            norm_num_groups=config.norm_num_groups,
            act_fn=act_fn(),
        )

        vq_embed_dim = config.vq_embed_dim if config.vq_embed_dim is not None else config.latents_channels

        self.quant_conv = nn.Conv2d(config.latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer(
            config.num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False
        )
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, config.latent_channels, 1)

        if config.speaker_embed_dim is not None:
            # Linear layer to project concatenated latents and speaker embeddings back to latent dimension
            self.speaker_latents_fc = nn.Linear(
                config.latent_channels + config.speaker_embed_dim,
                config.latent_channels
            )

        self.decoder = Decoder(
            output_channels=config.out_channels,
            num_layers=config.num_layers,
            initial_filters=config.initial_filters,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            norm_num_groups=config.norm_num_groups,
            act_fn=act_fn(),
        )

    def get_codebook_indices(self, x: torch.FloatTensor):
        x = x if len(x.shape) == 4 else x.view(x.shape[0], 1, x.shape[1], x.shape[2]) # transform to have 1 channel
        z_e = self.encoder(x)
        z_e = self.quant_conv(z_e)
        z_q, vq_loss, (perplexity, min_encodings, min_encoding_indices) = self.quantize(z_e)
        min_encoding_indices = min_encoding_indices.reshape(x.shape[0], -1) # add batch dim
        return min_encoding_indices, vq_loss

    def forward(
            self,
            input_ids: torch.FloatTensor,
            label_ids: torch.FloatTensor = None,
            speaker_embeddings: torch.FloatTensor = None,
            attention_masks: torch.FloatTensor = None
    ):
        z_e = self.encoder(input_ids)
        # print(f"z_e shape: {z_e.shape}")
        z_e = self.quant_conv(z_e)
        # print(f"z_e quant conved shape: {z_e.shape}")
        z_q, vq_loss, (perplexity, min_encodings, min_encoding_indices) = self.quantize(z_e)
        # print(f"z_q shape: {z_e.shape}")
        z_q = self.post_quant_conv(z_q)
        # print(f"z_q post shape: {z_q.shape}")

        if speaker_embeddings is not None and self.config.speaker_embed_dim is not None:
            # Ensure the speaker embeddings are broadcasted to match z_q dimensions
            speaker_embed = speaker_embeddings.unsqueeze(2).unsqueeze(3).expand(-1, -1, z_q.size(2), z_q.size(3))
            # Concatenate speaker embeddings with latents
            concat_z = torch.cat((z_q, speaker_embed), dim=1)
            # print(f"z_q concat shape: {concat_z.shape}")
            # Project concatenated tensor back to latent dimension
            bsz, channels, height, width = concat_z.shape
            concat_z = concat_z.view(bsz, channels, -1).permute(0, 2, 1).contiguous()
            concat_z = self.speaker_latents_fc(concat_z)
            concat_z = concat_z.permute(0, 2, 1).contiguous().view(bsz, -1, height, width)
            z_q = concat_z
            # print(f"z_q fc shape: {z_q.shape}")

        z_recon = self.decoder(z_q)
        # print(f"z_recon shape: {z_recon.shape}")

        if attention_masks is not None:
            # Compute the masked reconstruction loss
            recon_loss = torch.nn.functional.mse_loss(
                z_recon * attention_masks,
                input_ids * attention_masks,
                reduction='sum'
            )
            # print(f"recon_loss_first: {recon_loss}")
            # Normalize the loss by the number of valid (unmasked) elements
            recon_loss = recon_loss / attention_masks.sum()
            # print(f"recon_loss after div: {recon_loss}")
        else:
            recon_loss = torch.nn.functional.mse_loss(z_recon, input_ids)

        # print(f"vq_loss: {vq_loss}, recon_loss: {recon_loss}")

        loss = recon_loss + vq_loss

        # return {
        #     "loss": loss,
        #     "perplexity": perplexity,
        #     "min_encodings": min_encodings,
        #     "min_encoding_indices": min_encoding_indices,
        #     "reconstruction": z_recon
        # }
        return loss, z_recon