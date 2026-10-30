from types import SimpleNamespace

from torch import nn


def build_vit_encoder(
    image_size: int = 224,
    patch_size: int = 14,
    hidden_size: int = 192,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 3,
    intermediate_size: int = 768,
):
    """Build a ViT encoder from transformers without pretrained weights."""

    from transformers import ViTConfig, ViTModel

    config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        qkv_bias=True,
    )
    return ViTModel(config)


def infer_encoder_dim(encoder: nn.Module, fallback: int = 192) -> int:
    config = getattr(encoder, "config", SimpleNamespace(hidden_size=fallback))
    return int(getattr(config, "hidden_size", fallback))
