from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO detector dataset where train uses all images and val is a random labeled subset."
    )
    parser.add_argument(
        "--source-detector",
        type=Path,
        default=Path("outputs/dataset_merged/detector"),
        help="Existing YOLO detector dataset with images/{train,val} and labels/{train,val}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_detector_trainall_randomval"),
        help="Output detector dataset directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Random validation image ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for validation sampling.")
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it already exists.")
    return parser.parse_args()


def ensure_empty_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; pass --force to replace it.")
        resolved = path.resolve()
        cwd = Path.cwd().resolve()
        if cwd not in resolved.parents:
            raise ValueError(f"Refusing to remove output outside workspace: {resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True)


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def collect_images(source_detector: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for split in ("train", "val"):
        image_dir = source_detector / "images" / split
        if not image_dir.exists():
            continue
        for path in image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images[path.stem] = path
    return dict(sorted(images.items()))


def collect_labels(source_detector: Path) -> dict[str, Path]:
    labels: dict[str, Path] = {}
    for split in ("train", "val"):
        label_dir = source_detector / "labels" / split
        if not label_dir.exists():
            continue
        for path in label_dir.glob("*.txt"):
            labels[path.stem] = path
    return dict(sorted(labels.items()))


def write_dataset_yaml(output: Path, names: dict[int, str]) -> None:
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
        "nc": len(names),
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_detector = args.source_detector.resolve()
    output = args.output.resolve()
    if not 0 < args.val_ratio <= 1:
        raise ValueError("--val-ratio must be in (0, 1].")
    if not (source_detector / "data.yaml").exists():
        raise FileNotFoundError(f"Missing source data.yaml: {source_detector / 'data.yaml'}")

    source_yaml = yaml.safe_load((source_detector / "data.yaml").read_text(encoding="utf-8"))
    names = source_yaml.get("names") or {0: "f1_car"}
    images = collect_images(source_detector)
    labels = collect_labels(source_detector)
    missing_labels = sorted(set(images) - set(labels))
    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise ValueError(f"{len(missing_labels)} images are missing labels, e.g. {preview}")

    ensure_empty_output(output, force=args.force)
    image_ids = sorted(images)
    rng = random.Random(args.seed)
    val_count = max(1, round(len(image_ids) * args.val_ratio))
    val_ids = set(rng.sample(image_ids, val_count))
    modes: dict[str, int] = {"hardlink": 0, "copy": 0}

    for image_id in image_ids:
        mode = link_or_copy(images[image_id], output / "images" / "train" / images[image_id].name)
        modes[mode] += 1
        mode = link_or_copy(labels[image_id], output / "labels" / "train" / labels[image_id].name)
        modes[mode] += 1
        if image_id in val_ids:
            mode = link_or_copy(images[image_id], output / "images" / "val" / images[image_id].name)
            modes[mode] += 1
            mode = link_or_copy(labels[image_id], output / "labels" / "val" / labels[image_id].name)
            modes[mode] += 1

    write_dataset_yaml(output, names)
    summary = {
        "source_detector": str(source_detector),
        "output": str(output),
        "train_images": len(image_ids),
        "val_images": len(val_ids),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "validation_overlaps_train": True,
        "materialization": modes,
        "data_yaml": str((output / "data.yaml").resolve()),
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
