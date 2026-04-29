from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from f1_recognition.classifier import build_eval_transform, classify_crops, load_classifier_checkpoint
from f1_recognition.io_utils import ensure_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEAM_COLORS = {
    "Ferrari": "#e10600",
    "Maclaren": "#ff8700",
    "McLaren": "#ff8700",
    "Redbull": "#1e41ff",
    "RedBull": "#1e41ff",
    "Mercedes": "#00d2be",
    "AstonMartin": "#006f62",
    "Alpine": "#2293d1",
    "Williams": "#00a3e0",
    "Haas": "#b6babd",
    "Audi": "#d5001c",
    "RacingBulls": "#6692ff",
    "Cadillac": "#c9a44c",
}
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect, classify, and visualize F1 cars on images.")
    parser.add_argument("--input", required=True, type=Path, help="Image file or directory.")
    parser.add_argument("--detector", required=True, type=Path, help="YOLO detector best.pt.")
    parser.add_argument("--classifier", required=True, type=Path, help="Classifier best_classifier.pt.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a 'visualize' folder beside the input image(s).",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=None, help="Optional YOLO inference image size.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively when input is a directory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def default_output_dir(input_path: Path) -> Path:
    resolved_input = input_path.resolve()
    if resolved_input.is_file():
        return resolved_input.parent / "visualize"
    return resolved_input / "visualize"


def is_inside_visualize_dir(path: Path) -> bool:
    return any(part.lower() == "visualize" for part in path.parts)


def find_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not is_inside_visualize_dir(path.relative_to(input_path))
    )


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def load_visual_font(image_width: int) -> ImageFont.ImageFont:
    font_size = max(22, min(72, round(image_width / 140)))
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), font_size)
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: str,
    image_width: int,
    font: ImageFont.ImageFont,
) -> None:
    padding_x = 8
    padding_y = 5
    text_w, text_h = text_size(draw, text, font)
    label_w = text_w + padding_x * 2
    label_h = text_h + padding_y * 2
    x, y = xy
    x = max(0, min(x, image_width - label_w))
    y = max(0, y - label_h - 4)
    draw.rounded_rectangle((x, y, x + label_w, y + label_h), radius=4, fill=color)
    draw.text((x + padding_x, y + padding_y - 1), text, fill="white", font=font)


def build_visual_label(classified: dict, detector_confidence: float) -> str:
    team = classified.get("team", "unknown_team")
    car_model = classified.get("car_model", "unknown_car")
    return f"Team: {team} | Car: {car_model} | Conf: {detector_confidence:.2f}"


def visualize_image(
    image_path: Path,
    detector,
    classifier,
    label_maps: dict,
    heads: list[str],
    classifier_transform,
    output_dir: Path,
    conf_thres: float,
    imgsz: int | None,
    device: torch.device,
) -> tuple[list[dict], Path]:
    base_image = Image.open(image_path).convert("RGB")
    visualized = base_image.copy()
    draw = ImageDraw.Draw(visualized)
    font = load_visual_font(visualized.width)

    predict_kwargs = {
        "source": str(image_path.resolve()),
        "conf": conf_thres,
        "verbose": False,
        "device": str(device),
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz
    detections = detector.predict(**predict_kwargs)
    if not detections:
        raise RuntimeError(f"Detector returned no result object for {image_path}")

    instances = []
    boxes = detections[0].boxes
    if boxes is not None:
        box_rows = []
        crops = []
        for index in range(len(boxes)):
            x1, y1, x2, y2 = [int(round(value)) for value in boxes.xyxy[index].cpu().tolist()]
            detector_confidence = round(float(boxes.conf[index].cpu().item()), 5)
            crops.append(base_image.crop((x1, y1, x2, y2)))
            box_rows.append((index, x1, y1, x2, y2, detector_confidence))
        classified_rows = classify_crops(
            crops=crops,
            model=classifier,
            label_maps=label_maps,
            heads=heads,
            transform=classifier_transform,
            device=device,
        )
        for (index, x1, y1, x2, y2, detector_confidence), classified in zip(
            box_rows,
            classified_rows,
            strict=True,
        ):
            team = classified.get("team", "")
            color = TEAM_COLORS.get(team, "#ff2d55")
            label = build_visual_label(classified, detector_confidence)

            draw.rectangle((x1, y1, x2, y2), outline=color, width=6)
            draw_label(draw, (x1, y1), label, color=color, image_width=visualized.width, font=font)

            instances.append(
                {
                    "image": str(image_path.resolve()),
                    "image_name": image_path.name,
                    "detection_index": index,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "detector_confidence": detector_confidence,
                    **classified,
                }
            )

    image_output = output_dir / f"{image_path.stem}_visualized.jpg"
    visualized.save(image_output, quality=95)
    return instances, image_output


def write_summary_csv(path: Path, rows: list[dict], heads: list[str]) -> None:
    fieldnames = [
        "image",
        "image_name",
        "detection_index",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "detector_confidence",
    ]
    for head in heads:
        fieldnames.extend([head, f"{head}_confidence"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    from ultralytics import YOLO

    args = build_parser().parse_args()
    device = torch.device(args.device)
    image_paths = find_images(args.input, recursive=args.recursive)
    if not image_paths:
        raise ValueError(f"No images found under {args.input}")
    output_dir = ensure_dir((args.output_dir or default_output_dir(args.input)).resolve())

    detector = YOLO(str(args.detector.resolve()))
    loaded = load_classifier_checkpoint(args.classifier, device)
    classifier_transform = build_eval_transform(loaded.image_size)

    all_rows = []
    for image_index, image_path in enumerate(image_paths, start=1):
        instances, image_output = visualize_image(
            image_path=image_path,
            detector=detector,
            classifier=loaded.model,
            label_maps=loaded.label_maps,
            heads=loaded.heads,
            classifier_transform=classifier_transform,
            output_dir=output_dir,
            conf_thres=args.conf_thres,
            imgsz=args.imgsz,
            device=device,
        )
        all_rows.extend(instances)
        print(f"[{image_index}/{len(image_paths)}] {image_path.name}: {len(instances)} cars -> {image_output}")

    summary_path = output_dir / "predictions_summary.csv"
    write_summary_csv(summary_path, all_rows, heads=loaded.heads)
    print(f"Summary CSV written to {summary_path}")


if __name__ == "__main__":
    main()
