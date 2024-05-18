from typing import Union, Optional, List

from transformers import PreTrainedTokenizerFast, TensorType, BatchEncoding
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput, TruncationStrategy, \
    TextInputPair, PreTokenizedInputPair, EncodedInputPair
from transformers.utils import PaddingStrategy

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class XttsTokenizer(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            vocab_file,
            sep_token="[SEP]",
            cls_token="[CLS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            mask_token="[MASK]",
            **kwargs,
    ):
        print(vocab_file)
        kwargs.update(dict(
            tokenizer_file=vocab_file,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
        ))

        super().__init__(
            **kwargs
        )

    def encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            lang: Optional[str] = None,
            **kwargs,
    ) -> BatchEncoding:

        if lang is not None:
            if isinstance(text, str):
                text = f"[{lang}] {text}"
            if isinstance(text, list):
                text = [f"[{lang}] {item}" if isinstance(item, str) else item for item in text]

            if isinstance(text_pair, str):
                text_pair = f"[{lang}] {text_pair}"
            if isinstance(text, list):
                text_pair = [f"[{lang}] {item}" if isinstance(item, str) else item for item in text_pair]

        return super().encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
                List[PreTokenizedInputPair],
                List[EncodedInput],
                List[EncodedInputPair],
            ],
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            lang: Optional[str] = None,
            **kwargs,
    ) -> BatchEncoding:

        if lang is not None:
            batch_text_or_text_pairs = [
                f"[{lang}] {item}" if isinstance(item, str) else item for item in
                batch_text_or_text_pairs
            ]

        return super().batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
