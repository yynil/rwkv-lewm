import argparse
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download


DATASETS = {
    "pusht": {
        "repo": "quentinll/lewm-pusht",
        "file": "pusht_expert_train.h5.zst",
        "outputs": ["pusht_expert_train.h5"],
        "size": "13.1 GB compressed",
    },
    "tworoom": {
        "repo": "quentinll/lewm-tworooms",
        "file": "tworoom.tar.zst",
        "outputs": ["tworoom.h5"],
        "size": "3.43 GB compressed",
    },
    "cube": {
        "repo": "quentinll/lewm-cube",
        "file": "cube_single_expert.tar.zst",
        "outputs": ["ogbench/cube_single_expert.h5", "cube_single_expert.h5"],
        "size": "46.2 GB compressed",
    },
    "reacher": {
        "repo": "quentinll/lewm-reacher",
        "file": "reacher.tar.zst",
        "outputs": ["reacher.h5", "dmc/reacher_random.h5"],
        "size": "23.8 GB compressed",
    },
}


def run(cmd: list[str], cwd: Path | None = None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def decompress(archive: Path, out_dir: Path):
    if archive.name.endswith(".tar.zst"):
        run(["tar", "--zstd", "-xvf", str(archive), "-C", str(out_dir)])
    elif archive.name.endswith(".h5.zst"):
        target = out_dir / archive.name.removesuffix(".zst")
        run(["zstd", "-d", "-f", str(archive), "-o", str(target)])
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def main():
    parser = argparse.ArgumentParser(description="Download official LeWM datasets from HF.")
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=[*DATASETS.keys(), "all"],
        help="Dataset(s) to download. Start with tworoom or pusht before all.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable-wm")),
        help="Output root. Defaults to $STABLEWM_HOME or ~/.stable-wm.",
    )
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    names = list(DATASETS) if "all" in args.datasets else args.datasets
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"Output root: {output}")
    for name in names:
        spec = DATASETS[name]
        print(f"\n== {name}: {spec['repo']} / {spec['file']} ({spec['size']})")
        archive = Path(
            hf_hub_download(
                repo_id=spec["repo"],
                repo_type="dataset",
                filename=spec["file"],
                cache_dir=args.cache_dir,
            )
        )
        print(f"Downloaded archive: {archive}")
        decompress(archive, output)

        found = False
        for rel in spec["outputs"]:
            if (output / rel).exists():
                print(f"Found expected file: {output / rel}")
                found = True
        if not found:
            print("Warning: expected output name not found; inspect extracted files:")
            for path in sorted(output.rglob("*.h5")):
                print(" ", path)

        if not args.keep_archives and archive.is_file() and archive.parent != output:
            # Leave HF cache intact. Only remove archives explicitly downloaded into output/cache.
            pass

    if shutil.which("du"):
        run(["du", "-sh", str(output)])


if __name__ == "__main__":
    main()
