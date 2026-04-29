from __future__ import annotations

from collections import Counter, defaultdict


def accuracy(items: list[dict], truth_key: str, pred_key: str) -> float:
    if not items:
        return 0.0
    correct = sum(1 for item in items if item[truth_key] == item[pred_key])
    return correct / len(items)


def build_confusion(items: list[dict], truth_key: str, pred_key: str) -> dict[str, dict[str, int]]:
    labels = sorted({item[truth_key] for item in items} | {item[pred_key] for item in items})
    matrix = {truth: {pred: 0 for pred in labels} for truth in labels}
    for item in items:
        matrix[item[truth_key]][item[pred_key]] += 1
    return matrix


def subgroup_accuracy(items: list[dict], truth_key: str, pred_key: str, subgroup_key: str) -> dict[str, float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[str(item[subgroup_key])].append(item)
    return {group: accuracy(rows, truth_key, pred_key) for group, rows in grouped.items()}


def top_driver_confusions_by_team(items: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        if item["driver_true"] != item["driver_pred"]:
            key = f"{item['driver_true']} -> {item['driver_pred']}"
            grouped[item["team_true"]][key] += 1
    payload: dict[str, list[dict]] = {}
    for team, counts in grouped.items():
        payload[team] = [
            {"pair": pair, "count": count}
            for pair, count in counts.most_common(10)
        ]
    return payload
