import argparse
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=str(Path.home() / ".stable-wm"),
        help="Directory containing tworoom.h5.",
    )
    args = parser.parse_args()

    path = Path(args.data_root).expanduser() / "tworoom.h5"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    print(f"File: {path}")
    print(f"Size: {path.stat().st_size / 1024**3:.2f} GiB")
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            obj = f[key]
            if hasattr(obj, "shape"):
                print(f"{key:20s} shape={str(obj.shape):24s} dtype={obj.dtype}")


if __name__ == "__main__":
    main()
