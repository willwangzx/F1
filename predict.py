from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from f1_recognition.classifier import build_eval_transform, classify_crops, load_classifier_checkpoint
from f1_recognition.io_utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end F1 detection and classification on a single image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--detector", required=True, type=Path, help="YOLO detector checkpoint")
    parser.add_argument("--classifier", required=True, type=Path, help="best_classifier.pt checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=None, help="Optional YOLO inference image size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    from ultralytics import YOLO

    args = build_parser().parse_args()
    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir.resolve())

    detector = YOLO(str(args.detector.resolve()))
    loaded = load_classifier_checkpoint(args.classifier, device)
    classifier_transform = build_eval_transform(loaded.image_size)

    base_image = Image.open(args.image).convert("RGB")
    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    predict_kwargs = {
        "source": str(args.image.resolve()),
        "conf": args.conf_thres,
        "verbose": False,
        "device": args.device,
    }
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz
    detections = detector.predict(**predict_kwargs)
    if not detections:
        raise RuntimeError("Detector returned no results object")

    instances: list[dict] = []
    boxes = detections[0].boxes
    if boxes is not None:
        box_rows = []
        crops = []
        for index in range(len(boxes)):
            xyxy = boxes.xyxy[index].cpu().tolist()
            confidence = float(boxes.conf[index].cpu().item())
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
            crops.append(base_image.crop((x1, y1, x2, y2)))
            box_rows.append((x1, y1, x2, y2, confidence))
        classified_rows = classify_crops(
            crops=crops,
            model=loaded.model,
            label_maps=loaded.label_maps,
            heads=loaded.heads,
            transform=classifier_transform,
            device=device,
        )
        for (x1, y1, x2, y2, confidence), classified in zip(box_rows, classified_rows, strict=True):
            instance = {"bbox": [x1, y1, x2, y2], "detector_confidence": round(confidence, 5), **classified}
            instances.append(instance)
            label = " | ".join([*(instance[head] for head in loaded.heads), f"{confidence:.2f}"])
            draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
            draw.text((x1 + 4, max(0, y1 - 18)), label, fill="yellow")

    json_path = output_dir / f"{args.image.stem}_predictions.json"
    image_path = output_dir / f"{args.image.stem}_predictions.jpg"
    image.save(image_path, quality=95)
    json_path.write_text(json.dumps({"image": str(args.image.resolve()), "detections": instances}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved predictions to {json_path}")
    print(f"Saved visualization to {image_path}")


if __name__ == "__main__":
    main()
