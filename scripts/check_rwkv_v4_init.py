import argparse
import sys
from pathlib import Path

import hydra
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_lewm.modules import AdaLNZeroModulation
from train import build_model


def load_cfg(config_name: str, overrides: list[str]):
    config_path = Path(__file__).resolve().parents[1] / "config" / "train"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_path)):
        return hydra.compose(config_name=config_name, overrides=overrides)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pusht_rwkv_v4_stage2_ds")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_cfg(args.config_name, args.overrides)
    model = build_model(cfg)
    dynamics = model.dynamics

    print("config", args.config_name)
    print("embed_dim", cfg.model.embed_dim)
    print("dynamics_dim", cfg.model.dynamics_dim)
    print("token_fusion", cfg.model.token_fusion, type(model.fuse).__name__)
    print("input_proj", type(dynamics.input_proj).__name__)
    print("modulate_v_first", getattr(dynamics.cfg, "modulate_v_first", False))

    if cfg.model.dynamics_dim != cfg.model.embed_dim:
        raise AssertionError("v4 expects dynamics_dim == embed_dim")
    if type(model.fuse).__name__ != "Identity":
        raise AssertionError("v4 expects latent_identity token fusion")
    if type(dynamics.input_proj).__name__ != "Identity":
        raise AssertionError("v4 expects identity dynamics input projection")

    max_param_abs = 0.0
    max_output_abs = 0.0
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, AdaLNZeroModulation):
            continue
        count += 1
        module = module.float()
        param_abs = module.zero_init_max_abs()
        cond_dim = module.net[-1].in_features
        cond = torch.randn(2, 3, cond_dim)
        output_abs = max(chunk.detach().abs().max().item() for chunk in module(cond))
        print(f"{name}: chunks={module.num_chunks} param_max_abs={param_abs:.3e} output_max_abs={output_abs:.3e}")
        max_param_abs = max(max_param_abs, param_abs)
        max_output_abs = max(max_output_abs, output_abs)

    if count == 0:
        raise AssertionError("No AdaLNZeroModulation modules found")
    if max_param_abs != 0.0 or max_output_abs != 0.0:
        raise AssertionError(
            f"AdaLN-zero init failed: param_max_abs={max_param_abs}, output_max_abs={max_output_abs}"
        )
    print(f"checked {count} AdaLNZeroModulation modules; all zero initialized")


if __name__ == "__main__":
    main()
