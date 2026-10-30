"""Small compatibility helpers copied from LeWM's training utilities."""

import numpy as np
import torch


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    from stable_pretraining import data as dt

    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    from stable_pretraining import data as dt

    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone().clamp_min(1e-6)

    def norm_fn(x):
        return ((x - mean) / std).float()

    return dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
