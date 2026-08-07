from contextlib import contextmanager

import torch


@contextmanager
def tensor_text_features(text_encoder):
    r"""
    Makes `text_encoder.get_text_features` return a plain tensor for the duration of the context.

    `transformers >= 5` returns a `BaseModelOutputWithPooling`, while `MusicLDMPipeline._encode_prompt`
    (and any code following the same contract) expects the tensor that earlier versions returned.
    `MusicLDMPipeline` is deprecated upstream (`_last_supported_version = "0.33.1"`), so this shim
    patches the call site instead of the library.
    """
    was_patched = "get_text_features" in text_encoder.__dict__
    original_get_text_features = text_encoder.get_text_features

    def get_text_features(*args, **kwargs):
        text_features = original_get_text_features(*args, **kwargs)
        if not torch.is_tensor(text_features):
            text_features = text_features.pooler_output
        return text_features

    text_encoder.get_text_features = get_text_features
    try:
        yield
    finally:
        if was_patched:
            text_encoder.get_text_features = original_get_text_features
        else:
            del text_encoder.get_text_features
