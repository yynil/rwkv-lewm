import torch

from rwkv_lewm.losses import rwkv_lewm_loss
from rwkv_lewm.model import DummyVisionEncoder, RWKVLeWorldModel
from rwkv_lewm.modules import SIGReg


def test_forward_backward_cpu():
    image_shape = (3, 32, 32)
    model = RWKVLeWorldModel.from_components(
        encoder=DummyVisionEncoder(image_shape=image_shape, hidden_dim=32),
        encoder_dim=32,
        action_dim=4,
        embed_dim=32,
        dynamics_dim=64,
        dynamics_layers=2,
        dropout=0.0,
        backend="fallback",
    )
    batch = {
        "pixels": torch.randn(2, 8, *image_shape),
        "action": torch.randn(2, 8, 4),
    }
    metrics = rwkv_lewm_loss(model, batch, SIGReg(knots=5, num_proj=8), sigreg_weight=0.09)
    metrics["loss"].backward()
    assert torch.isfinite(metrics["loss"])
