from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch rename image files and save an old/new path mapping.")
    parser.add_argument("--image-dir", type=Path, default=Path("F1"), help="Directory that contains raw images.")
    parser.add_argument("--prefix", default="f1_", help="Prefix for renamed files.")
    parser.add_argument("--start", type=int, default=1, help="First numeric id.")
    parser.add_argument("--digits", type=int, default=5, help="Zero padding width.")
    parser.add_argument("--recursive", action="store_true", help="Rename images in subdirectories too.")
    parser.add_argument("--mapping", type=Path, default=Path("outputs/rename_mapping.csv"), help="CSV mapping output path.")
    parser.add_argument(
        "--update-csv",
        type=Path,
        default=None,
        help="Optional annotation CSV to rewrite image_path values with renamed paths.",
    )
    parser.add_argument(
        "--updated-csv",
        type=Path,
        default=None,
        help="Output path for the rewritten annotation CSV. Defaults to '<input>.renamed.csv'.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually rename files. Without this, only preview.")
    return parser


def find_images(image_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        (
            path
            for path in image_dir.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


def build_mapping(images: list[Path], prefix: str, start: int, digits: int) -> list[tuple[Path, Path]]:
    mapping = []
    for offset, old_path in enumerate(images):
        number = start + offset
        new_name = f"{prefix}{number:0{digits}d}{old_path.suffix.lower()}"
        mapping.append((old_path, old_path.with_name(new_name)))
    return mapping


def validate_mapping(mapping: list[tuple[Path, Path]]) -> None:
    old_paths = {old_path.resolve() for old_path, _ in mapping}
    new_paths = [new_path.resolve() for _, new_path in mapping]
    duplicate_targets = sorted({path for path in new_paths if new_paths.count(path) > 1})
    if duplicate_targets:
        details = ", ".join(str(path) for path in duplicate_targets[:5])
        raise ValueError(f"Duplicate rename targets: {details}")

    blocking_targets = [path for path in new_paths if path.exists() and path not in old_paths]
    if blocking_targets:
        details = ", ".join(str(path) for path in blocking_targets[:5])
        raise FileExistsError(f"Target files already exist: {details}")


def write_mapping(mapping_path: Path, mapping: list[tuple[Path, Path]], applied: bool) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["old_path", "new_path", "old_name", "new_name", "applied"])
        writer.writeheader()
        for old_path, new_path in mapping:
            writer.writerow(
                {
                    "old_path": str(old_path.resolve()),
                    "new_path": str(new_path.resolve()),
                    "old_name": old_path.name,
                    "new_name": new_path.name,
                    "applied": int(applied),
                }
            )


def rename_files(mapping: list[tuple[Path, Path]]) -> None:
    temp_mapping = []
    for index, (old_path, _) in enumerate(mapping):
        temp_path = old_path.with_name(f".rename_tmp_{index:06d}{old_path.suffix.lower()}")
        old_path.rename(temp_path)
        temp_mapping.append((temp_path, mapping[index][1]))

    for temp_path, new_path in temp_mapping:
        temp_path.rename(new_path)


def update_annotation_csv(csv_path: Path, updated_csv_path: Path, mapping: list[tuple[Path, Path]]) -> int:
    path_lookup = {str(old_path.resolve()): str(new_path.resolve()) for old_path, new_path in mapping}
    name_lookup = {old_path.name: str(new_path.resolve()) for old_path, new_path in mapping}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain an image_path column")
        rows = list(reader)
        fieldnames = reader.fieldnames

    updated_count = 0
    for row in rows:
        image_path = Path(row["image_path"])
        resolved = str(image_path.resolve()) if image_path.is_absolute() else ""
        replacement = path_lookup.get(resolved) or name_lookup.get(image_path.name)
        if replacement:
            row["image_path"] = replacement
            updated_count += 1

    updated_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with updated_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return updated_count


def main() -> None:
    args = build_parser().parse_args()
    image_dir = args.image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = find_images(image_dir, recursive=args.recursive)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    mapping = build_mapping(images, prefix=args.prefix, start=args.start, digits=args.digits)
    validate_mapping(mapping)

    print(f"Found {len(mapping)} images in {image_dir}")
    print("Preview:")
    for old_path, new_path in mapping[:10]:
        print(f"  {old_path.name} -> {new_path.name}")
    if len(mapping) > 10:
        print(f"  ... {len(mapping) - 10} more")

    if args.apply:
        rename_files(mapping)
        print("Renamed files.")
    else:
        print("Dry run only. Add --apply to rename files.")

    write_mapping(args.mapping, mapping, applied=args.apply)
    print(f"Mapping written to {args.mapping.resolve()}")

    if args.update_csv:
        updated_csv = args.updated_csv or args.update_csv.with_name(f"{args.update_csv.stem}.renamed{args.update_csv.suffix}")
        updated_count = update_annotation_csv(args.update_csv, updated_csv, mapping)
        print(f"Updated {updated_count} image_path values in {updated_csv.resolve()}")


if __name__ == "__main__":
    main()


