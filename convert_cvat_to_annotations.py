from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


FIELDNAMES = [
    "image_path",
    "x1",
    "y1",
    "x2",
    "y2",
    "team",
    "driver",
    "car_model",
    "split",
    "source",
    "batch_id",
    "tcam_visible",
    "tcam_color",
    "tcam_x1",
    "tcam_y1",
    "tcam_x2",
    "tcam_y2",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CVAT image XML annotations to this project's annotations CSV.")
    parser.add_argument("--xml", type=Path, default=Path("dataset/annotations.xml"), help="CVAT XML export path.")
    parser.add_argument("--image-root", type=Path, default=Path("F1"), help="Directory containing the annotated images.")
    parser.add_argument("--output", type=Path, default=Path("dataset/annotations.csv"), help="Output CSV path.")
    parser.add_argument("--label", default="f1_car", help="CVAT label name to convert.")
    parser.add_argument("--source", default="cvat", help="Value written to the source column.")
    parser.add_argument("--batch-prefix", default="cvat", help="Prefix for generated batch_id values.")
    return parser


def attribute_map(box: ET.Element) -> dict[str, str]:
    return {
        attribute.get("name", ""): (attribute.text or "").strip()
        for attribute in box.findall("attribute")
    }


def fmt_coord(value: str) -> str:
    return str(round(float(value), 2))


def convert(xml_path: Path, image_root: Path, label: str, source: str, batch_prefix: str) -> list[dict[str, str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows: list[dict[str, str]] = []
    missing_images: list[Path] = []

    for image in root.findall("image"):
        image_name = image.get("name")
        if not image_name:
            continue

        image_path = (image_root / image_name).resolve()
        if not image_path.exists():
            missing_images.append(image_path)

        for box_index, box in enumerate(image.findall("box")):
            if box.get("label") != label:
                continue
            attrs = attribute_map(box)
            rows.append(
                {
                    "image_path": str(image_path),
                    "x1": fmt_coord(box.attrib["xtl"]),
                    "y1": fmt_coord(box.attrib["ytl"]),
                    "x2": fmt_coord(box.attrib["xbr"]),
                    "y2": fmt_coord(box.attrib["ybr"]),
                    "team": attrs.get("team", "unknown_team"),
                    "driver": attrs.get("driver", "unknown_driver") or "unknown_driver",
                    "car_model": attrs.get("car_model", "unknown_car_model"),
                    "split": "",
                    "source": source,
                    "batch_id": f"{batch_prefix}_{Path(image_name).stem}",
                    "tcam_visible": "no",
                    "tcam_color": "",
                    "tcam_x1": "",
                    "tcam_y1": "",
                    "tcam_x2": "",
                    "tcam_y2": "",
                }
            )

    if missing_images:
        examples = ", ".join(str(path) for path in missing_images[:5])
        raise FileNotFoundError(f"Images referenced by CVAT XML were not found: {examples}")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    rows = convert(
        xml_path=args.xml,
        image_root=args.image_root,
        label=args.label,
        source=args.source,
        batch_prefix=args.batch_prefix,
    )
    if not rows:
        raise ValueError(f"No '{args.label}' boxes found in {args.xml}")

    write_csv(args.output, rows)
    image_count = len({row["image_path"] for row in rows})
    print(f"Converted {len(rows)} boxes from {image_count} images")
    print(f"CSV written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
