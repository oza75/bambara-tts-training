from transformers import PretrainedConfig


class XttsConfig(PretrainedConfig):
    model_type = "xtts"

    def __init__(
            self,
            enable_redaction: bool = False,
            kv_cache: bool = True,
            debug_loading_failures: bool = False,

            gpt_checkpoint: str = None,
            clvp_checkpoint: str = None,
            decoder_checkpoint: str = None,

            tokenizer_file: str = "vocab.json",
            dvae_checkpoint: str = "dvae.pth",
            xtts_checkpoint: str = "model.pth",
            mel_norm_file: str = "mel_stats.pth",

            num_chars: int = 255,
            max_text_length: int = 200,
            max_wav_length: int = 255995,  # ~11.6 seconds
            min_conditioning_length: int = 66150,  # 3 secs
            max_conditioning_length: int = 132300,  # 6 secs

            gpt_batch_size: int = 1,
            gpt_max_audio_tokens: int = 605,
            gpt_max_text_tokens: int = 402,
            gpt_max_prompt_tokens: int = 70,
            gpt_layers: int = 30,
            gpt_n_model_channels: int = 1024,
            gpt_n_heads: int = 16,
            gpt_number_text_tokens: int = None,
            gpt_start_text_token: int = None,
            gpt_stop_text_token: int = None,
            gpt_num_audio_tokens: int = 1026,
            gpt_start_audio_token: int = 1024,
            gpt_stop_audio_token: int = 1025,
            gpt_code_stride_len: int = 1024,
            gpt_loss_text_ce_weight: float = 0.01,
            gpt_loss_mel_ce_weight: float = 1.0,
            gpt_use_masking_gt_prompt_approach: bool = True,
            gpt_use_perceiver_resampler: bool = True,

            input_sample_rate: int = 22050,
            output_sample_rate: int = 24000,
            output_hop_length: int = 256,
            decoder_input_dim: int = 1024,
            d_vector_dim: int = 512,
            cond_d_vector_in_each_upsampling_layer: bool = True,

            # constants
            duration_const: int = 102400,
            **kwargs
    ):
        """A config class to represent XTTS model arguments that define the model structure.

           Args:
               gpt_batch_size (int): The size of the auto-regressive batch.
               enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
               kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
               gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
               clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
               decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
               num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

               For GPT model:
               gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
               gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
               gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
               gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
               gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
               gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
               gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
               gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
               gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
               gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
               gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
               gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
               gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
           """

        self.enable_redaction = enable_redaction
        self.kv_cache = kv_cache
        self.debug_loading_failures = debug_loading_failures

        self.gpt_checkpoint = gpt_checkpoint
        self.clvp_checkpoint = clvp_checkpoint
        self.decoder_checkpoint = decoder_checkpoint
        self.tokenizer_file = tokenizer_file
        self.dvae_checkpoint = dvae_checkpoint
        self.xtts_checkpoint = xtts_checkpoint
        self.mel_norm_file = mel_norm_file

        self.num_chars = num_chars
        self.max_text_length = max_text_length
        self.max_wav_length = max_wav_length  # ~11.6 seconds
        self.min_conditioning_length = min_conditioning_length
        self.max_conditioning_length = max_conditioning_length

        # XTTS GPT Encoder params
        self.gpt_batch_size = gpt_batch_size
        self.gpt_max_audio_tokens = gpt_max_audio_tokens
        self.gpt_max_text_tokens = gpt_max_text_tokens
        self.gpt_max_prompt_tokens = gpt_max_prompt_tokens
        self.gpt_layers = gpt_layers
        self.gpt_n_model_channels = gpt_n_model_channels
        self.gpt_n_heads = gpt_n_heads
        self.gpt_number_text_tokens = gpt_number_text_tokens
        self.gpt_start_text_token = gpt_start_text_token
        self.gpt_stop_text_token = gpt_stop_text_token
        self.gpt_num_audio_tokens = gpt_num_audio_tokens
        self.gpt_start_audio_token = gpt_start_audio_token
        self.gpt_stop_audio_token = gpt_stop_audio_token
        self.gpt_code_stride_len = gpt_code_stride_len
        self.gpt_use_masking_gt_prompt_approach = gpt_use_masking_gt_prompt_approach
        self.gpt_use_perceiver_resampler = gpt_use_perceiver_resampler
        self.gpt_loss_text_ce_weight = gpt_loss_text_ce_weight
        self.gpt_loss_mel_ce_weight = gpt_loss_mel_ce_weight

        # HifiGAN Decoder params
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_hop_length = output_hop_length
        self.decoder_input_dim = decoder_input_dim
        self.d_vector_dim = d_vector_dim
        self.cond_d_vector_in_each_upsampling_layer = cond_d_vector_in_each_upsampling_layer
        self.duration_const = duration_const

        super().__init__(**kwargs)
