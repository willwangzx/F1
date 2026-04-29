from __future__ import annotations

import argparse
from pathlib import Path

from f1_recognition.io_utils import ensure_dir, read_csv, save_json
from f1_recognition.metrics import accuracy, build_confusion, subgroup_accuracy, top_driver_confusions_by_team

CLASSIFIER_HEADS = ("team", "driver", "car_model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze classifier evaluation output.")
    parser.add_argument("--eval-csv", required=True, type=Path, help="eval_classifier.csv from train_classifier.py")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_csv(args.eval_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.eval_csv}")

    output_dir = ensure_dir(args.output_dir.resolve())
    available_heads = [
        head
        for head in CLASSIFIER_HEADS
        if f"{head}_true" in rows[0] and f"{head}_pred" in rows[0]
    ]
    metrics = {f"{head}_accuracy": accuracy(rows, f"{head}_true", f"{head}_pred") for head in available_heads}
    if "driver" in available_heads:
        metrics["driver_accuracy_by_tcam_visible"] = subgroup_accuracy(
            rows,
            "driver_true",
            "driver_pred",
            "tcam_visible",
        )
        metrics["top_driver_confusions_by_team"] = top_driver_confusions_by_team(rows)
    confusions = {head: build_confusion(rows, f"{head}_true", f"{head}_pred") for head in available_heads}
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "confusions.json", confusions)

    report_lines = [
        "# F1 Classifier Error Analysis",
        "",
    ]
    for head in available_heads:
        report_lines.append(f"- {head.replace('_', ' ').title()} accuracy: {metrics[f'{head}_accuracy']:.3f}")
    if "driver" in available_heads:
        report_lines.extend(["", "## Driver Accuracy by T-cam Visibility"])
        for key, value in metrics["driver_accuracy_by_tcam_visible"].items():
            report_lines.append(f"- tcam_visible={key}: {value:.3f}")
        report_lines.extend(["", "## Top Driver Confusions by Team"])
        for team, entries in metrics["top_driver_confusions_by_team"].items():
            if not entries:
                continue
            report_lines.append(f"- {team}: " + ", ".join(f"{entry['pair']} ({entry['count']})" for entry in entries[:5]))

    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
