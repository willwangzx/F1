from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from shutil import copy2

from PIL import Image

from f1_recognition.annotations import load_annotations, summarize_annotations
from f1_recognition.io_utils import (
    clamp_bbox,
    ensure_dir,
    relative_to,
    save_json,
    save_yaml,
    split_groups,
    write_csv,
    resize_to_long_edge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare F1 detection and classification datasets from CSV annotations.")
    parser.add_argument("--annotations", required=True, type=Path, help="CSV annotations file.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional base directory for relative image paths.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset"), help="Dataset export directory.")
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=2560,
        help="Resize images so the long edge does not exceed this value. Use 0 to keep original pixels.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when split is not provided.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouped split assignment.")
    parser.add_argument("--crop-padding", type=float, default=0.05, help="Extra padding around the car crop as a bbox fraction.")
    return parser


def materialize_original_image(source: Path, target: Path) -> str:
    """Place source at target via hardlink when possible, falling back to copy."""
    ensure_dir(target.parent)
    if target.exists():
        target.unlink()
    try:
        target.hardlink_to(source)
        return "hardlink"
    except OSError:
        copy2(source, target)
        return "copy"


def assign_splits(rows: list[dict], val_ratio: float, seed: int) -> None:
    preset = any(row["split"] for row in rows)
    if preset:
        for row in rows:
            row["split"] = row["split"] or "train"
        return

    groups = []
    for row in rows:
        group = row["batch_id"] or Path(row["image_path"]).stem
        groups.append(group)
    train_groups, val_groups = split_groups(groups, val_ratio=val_ratio, seed=seed)
    for row, group in zip(rows, groups, strict=True):
        row["split"] = "val" if group in val_groups else "train"


def validate_split_consistency(rows: list[dict]) -> dict:
    split_by_batch: dict[str, set[str]] = defaultdict(set)
    missing_batch_ids = 0
    for row in rows:
        if row["batch_id"]:
            split_by_batch[row["batch_id"]].add(row["split"])
        else:
            missing_batch_ids += 1
    conflicting_batches = {
        batch_id: sorted(list(splits))
        for batch_id, splits in split_by_batch.items()
        if len(splits) > 1
    }
    if conflicting_batches:
        details = ", ".join(f"{batch_id}:{'/'.join(splits)}" for batch_id, splits in conflicting_batches.items())
        raise ValueError(f"Found batch_id values spanning multiple splits: {details}")
    return {
        "missing_batch_ids": missing_batch_ids,
        "conflicting_batch_ids": conflicting_batches,
    }


def main() -> None:
    args = build_parser().parse_args()
    rows = load_annotations(args.annotations, image_root=args.image_root)
    assign_splits(rows, val_ratio=args.val_ratio, seed=args.seed)
    validation_report = validate_split_consistency(rows)

    output_dir = args.output_dir.resolve()
    processed_dir = ensure_dir(output_dir / "processed_images")
    detector_images_dir = ensure_dir(output_dir / "detector" / "images")
    detector_labels_dir = ensure_dir(output_dir / "detector" / "labels")
    classifier_crops_dir = ensure_dir(output_dir / "classifier" / "crops")
    metadata_dir = ensure_dir(output_dir / "metadata")

    grouped_by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_by_image[row["image_path"]].append(row)

    team_names = sorted({row["team"] for row in rows})
    driver_names = sorted({row["driver"] for row in rows})
    car_model_names = sorted({row["car_model"] for row in rows})
    label_maps = {
        "team_to_id": {name: index for index, name in enumerate(team_names)},
        "driver_to_id": {name: index for index, name in enumerate(driver_names)},
        "car_model_to_id": {name: index for index, name in enumerate(car_model_names)},
        "team_names": team_names,
        "driver_names": driver_names,
        "car_model_names": car_model_names,
    }
    save_json(metadata_dir / "label_maps.json", label_maps)

    classifier_rows: list[dict] = []
    image_manifest_rows: list[dict] = []

    for image_path, image_rows in grouped_by_image.items():
        original_path = Path(image_path)
        image = Image.open(original_path).convert("RGB")
        original_width, original_height = image.size
        for row in image_rows:
            if not (0 <= row["x1"] < row["x2"] <= original_width and 0 <= row["y1"] < row["y2"] <= original_height):
                raise ValueError(
                    f"BBox out of bounds for {original_path.name}: "
                    f"({row['x1']}, {row['y1']}, {row['x2']}, {row['y2']}) "
                    f"vs image size ({original_width}, {original_height})"
                )
            tcam_values = [row["tcam_x1"], row["tcam_y1"], row["tcam_x2"], row["tcam_y2"]]
            if any(tcam_values):
                tx1, ty1, tx2, ty2 = [float(value) for value in tcam_values]
                if not (0 <= tx1 < tx2 <= original_width and 0 <= ty1 < ty2 <= original_height):
                    raise ValueError(
                        f"T-cam bbox out of bounds for {original_path.name}: "
                        f"({tx1}, {ty1}, {tx2}, {ty2}) vs image size ({original_width}, {original_height})"
                    )
        keep_original_pixels = args.max_long_edge <= 0
        if keep_original_pixels:
            resized_image = image.copy()
            scale = 1.0
        else:
            resized_image, scale = resize_to_long_edge(image, max_long_edge=args.max_long_edge)
        width, height = resized_image.size
        processed_name = original_path.name if keep_original_pixels else f"{original_path.stem}.jpg"
        processed_path = processed_dir / processed_name

        split = image_rows[0]["split"]
        detector_image_path = ensure_dir(detector_images_dir / split) / processed_name
        if keep_original_pixels:
            processed_mode = materialize_original_image(original_path, processed_path)
            detector_mode = materialize_original_image(original_path, detector_image_path)
        else:
            resized_image.save(processed_path, quality=95)
            resized_image.save(detector_image_path, quality=95)
            processed_mode = "encoded_jpeg"
            detector_mode = "encoded_jpeg"

        detector_label_lines: list[str] = []
        for index, row in enumerate(image_rows):
            x1, y1, x2, y2 = (
                row["x1"] * scale,
                row["y1"] * scale,
                row["x2"] * scale,
                row["y2"] * scale,
            )
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, width, height)
            xc = ((x1 + x2) / 2.0) / width
            yc = ((y1 + y2) / 2.0) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            detector_label_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            pad_w = (x2 - x1) * args.crop_padding
            pad_h = (y2 - y1) * args.crop_padding
            crop_x1, crop_y1, crop_x2, crop_y2 = clamp_bbox(x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h, width, height)
            crop = resized_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_dir = ensure_dir(classifier_crops_dir / split)
            crop_name = f"{original_path.stem}_{index:03d}.jpg"
            crop_path = crop_dir / crop_name
            crop.save(crop_path, quality=95)

            classifier_rows.append(
                {
                    "crop_path": str(crop_path.resolve()),
                    "split": split,
                    "image_path": str(processed_path.resolve()),
                    "image_name": processed_name,
                    "source": row["source"],
                    "batch_id": row["batch_id"],
                    "bbox_x1": crop_x1,
                    "bbox_y1": crop_y1,
                    "bbox_x2": crop_x2,
                    "bbox_y2": crop_y2,
                    "team": row["team"],
                    "driver": row["driver"],
                    "car_model": row["car_model"],
                    "team_id": label_maps["team_to_id"][row["team"]],
                    "driver_id": label_maps["driver_to_id"][row["driver"]],
                    "car_model_id": label_maps["car_model_to_id"][row["car_model"]],
                    "tcam_visible": int(row["tcam_visible"]),
                    "tcam_color": row["tcam_color"],
                    "tcam_x1": row["tcam_x1"],
                    "tcam_y1": row["tcam_y1"],
                    "tcam_x2": row["tcam_x2"],
                    "tcam_y2": row["tcam_y2"],
                }
            )

        label_path = ensure_dir(detector_labels_dir / split) / f"{original_path.stem}.txt"
        label_path.write_text("\n".join(detector_label_lines), encoding="utf-8")

        image_manifest_rows.append(
            {
                "image_path": str(processed_path.resolve()),
                "split": split,
                "original_path": str(original_path.resolve()),
                "box_count": len(image_rows),
                "width": width,
                "height": height,
                "scale": round(scale, 6),
                "processed_mode": processed_mode,
                "detector_mode": detector_mode,
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

    classifier_fieldnames = [
        "crop_path",
        "split",
        "image_path",
        "image_name",
        "source",
        "batch_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "team",
        "driver",
        "car_model",
        "team_id",
        "driver_id",
        "car_model_id",
        "tcam_visible",
        "tcam_color",
        "tcam_x1",
        "tcam_y1",
        "tcam_x2",
        "tcam_y2",
    ]
    write_csv(output_dir / "classifier" / "manifest.csv", classifier_rows, classifier_fieldnames)
    write_csv(output_dir / "metadata" / "images.csv", image_manifest_rows, list(image_manifest_rows[0].keys()))

    summary = summarize_annotations(rows)
    summary["split_counts"] = {
        "train_boxes": sum(1 for row in rows if row["split"] == "train"),
        "val_boxes": sum(1 for row in rows if row["split"] == "val"),
        "train_images": sum(1 for row in image_manifest_rows if row["split"] == "train"),
        "val_images": sum(1 for row in image_manifest_rows if row["split"] == "val"),
    }
    summary["paths"] = {
        "dataset_root": str(output_dir),
        "detector_yaml": str((output_dir / "detector" / "data.yaml").resolve()),
        "classifier_manifest": str((output_dir / "classifier" / "manifest.csv").resolve()),
        "label_maps": str((output_dir / "metadata" / "label_maps.json").resolve()),
    }
    summary["validation"] = validation_report
    save_json(output_dir / "metadata" / "dataset_summary.json", summary)

    print(f"Prepared dataset at {output_dir}")
    print(f"Detector YAML: {output_dir / 'detector' / 'data.yaml'}")
    print(f"Classifier manifest: {output_dir / 'classifier' / 'manifest.csv'}")


if __name__ == "__main__":
    main()
