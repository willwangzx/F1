from __future__ import annotations

from collections import Counter
from pathlib import Path

from .io_utils import parse_bool, read_csv

REQUIRED_COLUMNS = {
    "image_path",
    "x1",
    "y1",
    "x2",
    "y2",
    "team",
    "driver",
    "car_model",
}

OPTIONAL_COLUMNS = {
    "split",
    "source",
    "batch_id",
    "tcam_visible",
    "tcam_color",
    "tcam_x1",
    "tcam_y1",
    "tcam_x2",
    "tcam_y2",
}


def load_annotations(csv_path: Path, image_root: Path | None = None) -> list[dict]:
    raw_rows = read_csv(csv_path)
    if not raw_rows:
        raise ValueError(f"No annotations found in {csv_path}")
    headers = {header.strip() for header in raw_rows[0].keys() if header}
    missing = REQUIRED_COLUMNS - headers
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required annotation columns: {missing_list}")

    rows: list[dict] = []
    for index, raw in enumerate(raw_rows, start=2):
        image_value = (raw.get("image_path") or "").strip()
        if not image_value:
            raise ValueError(f"Row {index}: image_path is empty")
        image_path = Path(image_value)
        if not image_path.is_absolute():
            if image_root is None:
                image_path = (csv_path.parent / image_path).resolve()
            else:
                image_path = (image_root / image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Row {index}: image not found -> {image_path}")

        x1 = float(raw["x1"])
        y1 = float(raw["y1"])
        x2 = float(raw["x2"])
        y2 = float(raw["y2"])
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Row {index}: invalid bbox ({x1}, {y1}, {x2}, {y2})")

        split = (raw.get("split") or "").strip().lower()
        if split and split not in {"train", "val"}:
            raise ValueError(f"Row {index}: split must be train/val or blank")

        row = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "team": (raw["team"] or "").strip(),
            "driver": (raw["driver"] or "").strip(),
            "car_model": (raw["car_model"] or "").strip(),
            "split": split,
            "source": (raw.get("source") or "private").strip(),
            "batch_id": (raw.get("batch_id") or "").strip(),
            "tcam_visible": parse_bool(raw.get("tcam_visible"), default=False),
            "tcam_color": (raw.get("tcam_color") or "").strip(),
            "tcam_x1": (raw.get("tcam_x1") or "").strip(),
            "tcam_y1": (raw.get("tcam_y1") or "").strip(),
            "tcam_x2": (raw.get("tcam_x2") or "").strip(),
            "tcam_y2": (raw.get("tcam_y2") or "").strip(),
        }
        for key in ("team", "driver", "car_model"):
            if not row[key]:
                raise ValueError(f"Row {index}: {key} is empty")
        rows.append(row)
    return rows


def summarize_annotations(rows: list[dict]) -> dict:
    team_counts = Counter(row["team"] for row in rows)
    driver_counts = Counter(row["driver"] for row in rows)
    model_counts = Counter(row["car_model"] for row in rows)
    image_counts = Counter(Path(row["image_path"]).name for row in rows)
    return {
        "boxes": len(rows),
        "images": len(image_counts),
        "teams": dict(sorted(team_counts.items())),
        "drivers": dict(sorted(driver_counts.items())),
        "car_models": dict(sorted(model_counts.items())),
        "unknown_driver_boxes": driver_counts.get("unknown_driver", 0),
        "tcam_visible_boxes": sum(1 for row in rows if row["tcam_visible"]),
    }
