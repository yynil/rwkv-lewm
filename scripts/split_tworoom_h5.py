import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Split a stable-worldmodel HDF5 dataset by episode.")
    parser.add_argument("--input", default=str(Path.home() / ".stable-wm" / "tworoom.h5"))
    parser.add_argument("--output-dir", default=str(Path.home() / ".stable-wm"))
    parser.add_argument("--train-name", default="tworoom_train")
    parser.add_argument("--eval-name", default="tworoom_eval")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_episode_split(num_episodes: int, eval_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_episodes)
    eval_count = int(round(num_episodes * eval_ratio))
    eval_eps = np.sort(perm[:eval_count])
    train_eps = np.sort(perm[eval_count:])
    return train_eps, eval_eps


def create_dataset_like(dst, name: str, src_ds, rows: int):
    shape = (rows, *src_ds.shape[1:])
    chunks = src_ds.chunks
    if chunks is not None:
        chunks = (min(chunks[0], max(rows, 1)), *chunks[1:])
    return dst.create_dataset(
        name,
        shape=shape,
        dtype=src_ds.dtype,
        chunks=chunks,
    )


def copy_attrs(src_obj, dst_obj):
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def write_split(src_path: Path, dst_path: Path, episode_ids: np.ndarray, overwrite: bool):
    if dst_path.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_path} exists; pass --overwrite to replace it")
        dst_path.unlink()

    with h5py.File(src_path, "r") as src:
        lengths = src["ep_len"][:].astype(np.int64)
        offsets = src["ep_offset"][:].astype(np.int64)
        selected_lengths = lengths[episode_ids]
        total_rows = int(selected_lengths.sum())
        new_offsets = np.concatenate([[0], np.cumsum(selected_lengths[:-1])]).astype(np.int64)

        with h5py.File(dst_path, "w") as dst:
            copy_attrs(src, dst)
            dst.create_dataset("ep_len", data=selected_lengths.astype(src["ep_len"].dtype))
            dst.create_dataset("ep_offset", data=new_offsets.astype(src["ep_offset"].dtype))

            data_keys = [key for key in src.keys() if key not in ("ep_len", "ep_offset")]
            writers = {key: create_dataset_like(dst, key, src[key], total_rows) for key in data_keys}
            for key in data_keys:
                copy_attrs(src[key], writers[key])

            row = 0
            for new_ep, old_ep in enumerate(tqdm(episode_ids, desc=dst_path.name, dynamic_ncols=True)):
                start = int(offsets[old_ep])
                length = int(lengths[old_ep])
                end = start + length
                out = slice(row, row + length)

                for key in data_keys:
                    if key in ("ep_idx", "episode_idx"):
                        writers[key][out] = np.full(length, new_ep, dtype=writers[key].dtype)
                    elif key == "id":
                        writers[key][out] = np.arange(row, row + length, dtype=writers[key].dtype)
                    else:
                        writers[key][out] = src[key][start:end]
                row += length

            dst.attrs["source_file"] = str(src_path)
            dst.attrs["source_episode_ids"] = json.dumps([int(x) for x in episode_ids])


def inspect_file(path: Path):
    with h5py.File(path, "r") as f:
        ep_len = f["ep_len"][:]
        return {
            "path": str(path),
            "episodes": int(len(ep_len)),
            "rows": int(ep_len.sum()),
            "min_len": int(ep_len.min()),
            "max_len": int(ep_len.max()),
            "mean_len": float(ep_len.mean()),
        }


def main():
    args = parse_args()
    src_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{args.train_name}.h5"
    eval_path = out_dir / f"{args.eval_name}.h5"

    with h5py.File(src_path, "r") as f:
        num_episodes = len(f["ep_len"])

    train_eps, eval_eps = make_episode_split(num_episodes, args.eval_ratio, args.seed)
    print(f"source={src_path}")
    print(f"episodes={num_episodes} train={len(train_eps)} eval={len(eval_eps)} seed={args.seed}")

    write_split(src_path, train_path, train_eps, args.overwrite)
    write_split(src_path, eval_path, eval_eps, args.overwrite)

    summary = {
        "source": inspect_file(src_path),
        "train": inspect_file(train_path),
        "eval": inspect_file(eval_path),
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "tworoom_split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
