from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from shutil import copy2

from PIL import Image, ImageEnhance, ImageOps

from f1_recognition.annotations import load_annotations, summarize_annotations
from f1_recognition.io_utils import ensure_dir, save_json, save_yaml, split_groups, write_csv


AUGMENTATIONS = ("hflip", "bright", "contrast")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an expanded full-resolution detector dataset.")
    parser.add_argument("--annotations", required=True, type=Path, help="CSV annotations file.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional base directory for relative image paths.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_f1_augmented_original"))
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when split is not provided.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouped split assignment.")
    parser.add_argument(
        "--augmentations",
        nargs="+",
        choices=AUGMENTATIONS,
        default=list(AUGMENTATIONS),
        help="Offline train-only augmentations to add while preserving full image dimensions.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def assign_splits(rows: list[dict], val_ratio: float, seed: int) -> None:
    preset = any(row["split"] for row in rows)
    if preset:
        for row in rows:
            row["split"] = row["split"] or "train"
        return

    groups = [row["batch_id"] or Path(row["image_path"]).stem for row in rows]
    _, val_groups = split_groups(groups, val_ratio=val_ratio, seed=seed)
    for row, group in zip(rows, groups, strict=True):
        row["split"] = "val" if group in val_groups else "train"


def materialize_original_image(source: Path, target: Path) -> str:
    ensure_dir(target.parent)
    if target.exists():
        target.unlink()
    try:
        target.hardlink_to(source)
        return "hardlink"
    except OSError:
        copy2(source, target)
        return "copy"


def transform_boxes(rows: list[dict], variant: str, width: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    for row in rows:
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        if variant == "hflip":
            x1, x2 = width - x2, width - x1
        boxes.append((x1, y1, x2, y2))
    return boxes


def save_label(path: Path, boxes: list[tuple[float, float, float, float]], width: int, height: int) -> None:
    lines = []
    for x1, y1, x2, y2 in boxes:
        xc = ((x1 + x2) / 2.0) / width
        yc = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def apply_augmentation(image: Image.Image, variant: str) -> Image.Image:
    if variant == "hflip":
        return ImageOps.mirror(image)
    if variant == "bright":
        return ImageEnhance.Brightness(image).enhance(1.12)
    if variant == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.15)
    raise ValueError(f"Unsupported augmentation: {variant}")


def main() -> None:
    args = build_parser().parse_args()
    rows = load_annotations(args.annotations, image_root=args.image_root)
    assign_splits(rows, val_ratio=args.val_ratio, seed=args.seed)

    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "detector" / "images"
    labels_dir = output_dir / "detector" / "labels"
    metadata_dir = ensure_dir(output_dir / "metadata")

    grouped_by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_by_image[row["image_path"]].append(row)

    image_rows: list[dict] = []
    for image_path, image_annotations in grouped_by_image.items():
        original_path = Path(image_path)
        split = image_annotations[0]["split"]
        with Image.open(original_path) as image_file:
            width, height = image_file.size

        variants = ["original"]
        if split == "train":
            variants.extend(args.augmentations)

        for variant in variants:
            if variant == "original":
                output_name = original_path.name
                mode = materialize_original_image(original_path, images_dir / split / output_name)
            else:
                output_name = f"{original_path.stem}__aug_{variant}.jpg"
                with Image.open(original_path).convert("RGB") as image:
                    augmented = apply_augmentation(image, variant)
                    ensure_dir(images_dir / split)
                    augmented.save(images_dir / split / output_name, quality=args.jpeg_quality, subsampling=0)
                mode = "encoded_augmented_jpeg"

            label_name = f"{Path(output_name).stem}.txt"
            boxes = transform_boxes(image_annotations, variant=variant, width=width)
            save_label(labels_dir / split / label_name, boxes, width=width, height=height)
            image_rows.append(
                {
                    "image_path": str((images_dir / split / output_name).resolve()),
                    "split": split,
                    "original_path": str(original_path.resolve()),
                    "variant": variant,
                    "box_count": len(image_annotations),
                    "width": width,
                    "height": height,
                    "scale": 1.0,
                    "mode": mode,
                }
            )

    detector_yaml = {
        "path": str((output_dir / "detector").resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "f1_car"},
        "nc": 1,
    }
    save_yaml(output_dir / "detector" / "data.yaml", detector_yaml)
    write_csv(output_dir / "metadata" / "images.csv", image_rows, list(image_rows[0].keys()))

    summary = summarize_annotations(rows)
    summary["split_counts"] = {
        "train_images_original": sum(1 for row in image_rows if row["split"] == "train" and row["variant"] == "original"),
        "train_images_total": sum(1 for row in image_rows if row["split"] == "train"),
        "val_images": sum(1 for row in image_rows if row["split"] == "val"),
        "train_boxes_original": sum(1 for row in rows if row["split"] == "train"),
        "val_boxes": sum(1 for row in rows if row["split"] == "val"),
    }
    summary["augmentations"] = list(args.augmentations)
    summary["paths"] = {
        "dataset_root": str(output_dir),
        "detector_yaml": str((output_dir / "detector" / "data.yaml").resolve()),
    }
    save_json(output_dir / "metadata" / "dataset_summary.json", summary)

    print(f"Prepared augmented detector dataset at {output_dir}")
    print(f"Detector YAML: {output_dir / 'detector' / 'data.yaml'}")
    print(f"Train images: {summary['split_counts']['train_images_total']}")
    print(f"Val images: {summary['split_counts']['val_images']}")


if __name__ == "__main__":
    main()
