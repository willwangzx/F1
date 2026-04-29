from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def save_json(path: Path, payload: dict | list) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return default


def split_groups(
    group_ids: Iterable[str],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    unique_groups = sorted(set(group_ids))
    if not unique_groups:
        return set(), set()
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    val_count = max(1, int(round(len(unique_groups) * val_ratio))) if len(unique_groups) > 1 else 0
    val_groups = set(unique_groups[:val_count])
    train_groups = set(unique_groups[val_count:]) or (set(unique_groups) - val_groups)
    if not train_groups and val_groups:
        group = next(iter(val_groups))
        val_groups.remove(group)
        train_groups.add(group)
    return train_groups, val_groups


def resize_to_long_edge(image: Image.Image, max_long_edge: int) -> tuple[Image.Image, float]:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image.copy(), 1.0
    scale = max_long_edge / float(long_edge)
    resized = image.resize((int(round(width * scale)), int(round(height * scale))), Image.Resampling.LANCZOS)
    return resized, scale


def clamp_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return x1, y1, x2, y2


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
