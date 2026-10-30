import argparse

import torch

from rwkv_lewm.losses import rwkv_lewm_loss
from rwkv_lewm.model import DummyVisionEncoder, RWKVLeWorldModel
from rwkv_lewm.modules import SIGReg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", default="rwkv7_cuda", choices=["rwkv7_cuda", "fallback"])
    args = parser.parse_args()

    device = torch.device(args.device)
    image_shape = (3, 64, 64)
    embed_dim = 64
    action_dim = 10
    model = RWKVLeWorldModel.from_components(
        encoder=DummyVisionEncoder(image_shape=image_shape, hidden_dim=embed_dim),
        encoder_dim=embed_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        dynamics_dim=128,
        dynamics_layers=3,
        dropout=0.0,
        backend=args.backend,
    ).to(device)
    sigreg = SIGReg(knots=5, num_proj=16).to(device)

    batch = {
        "pixels": torch.randn(4, 16, *image_shape, device=device),
        "action": torch.randn(4, 16, action_dim, device=device),
    }
    metrics = rwkv_lewm_loss(model, batch, sigreg, sigreg_weight=0.09)
    metrics["loss"].backward()

    info = {
        "pixels": torch.randn(2, 5, 4, *image_shape, device=device),
        "goal": torch.randn(2, 5, 4, *image_shape, device=device),
        "action": torch.randn(2, 5, 4, action_dim, device=device),
    }
    action_candidates = torch.randn(2, 5, 8, action_dim, device=device)
    cost = model.get_cost(info, action_candidates)
    if device.type == "cuda":
        torch.cuda.synchronize()

    assert cost.shape == (2, 5)
    print("RWKV-LeWM smoke test passed")
    print("device", device)
    print("loss", float(metrics["loss"].detach().cpu()))
    print("pred_loss", float(metrics["pred_loss"].detach().cpu()))
    print("sigreg_loss", float(metrics["sigreg_loss"].detach().cpu()))
    print("cost", tuple(cost.shape), cost.device)


if __name__ == "__main__":
    main()
