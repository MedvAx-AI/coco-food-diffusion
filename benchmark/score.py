"""Summarize the fixed paired benchmark from saved DETR predictions."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ("pretrained", "tuned")
THRESHOLDS = (0.30, 0.50, 0.70)


def score(prompt, detections, threshold):
    foods = [d for d in detections if d["label"] == prompt["food"] and d["confidence"] >= threshold]
    if prompt["task"] == "identity":
        return bool(foods)
    if prompt["task"] == "count":
        return len(foods) == int(prompt["count"])
    cups = [d for d in detections if d["label"] == "cup" and d["confidence"] >= threshold]
    if not foods or not cups:
        return False
    food, cup = (max(group, key=lambda d: d["confidence"])["box"] for group in (foods, cups))
    delta = (food[0] + food[2] - cup[0] - cup[2]) / 2
    return delta < -8 if prompt["side"] == "left" else delta > 8


def summarize(prompts, data):
    rows = []
    for threshold in THRESHOLDS:
        for task in ("identity", "count", "spatial"):
            pairs = [
                tuple(score(prompts[s["id"]], s[model], threshold) for model in MODELS)
                for s in data["samples"] if prompts[s["id"]]["task"] == task
            ]
            rows.append(dict(
                threshold=threshold, task=task, pairs=len(pairs),
                pretrained=sum(a for a, _ in pairs), tuned=sum(b for _, b in pairs),
                tuned_only=sum(b and not a for a, b in pairs),
                pretrained_only=sum(a and not b for a, b in pairs),
            ))
    return rows


def main():
    with (ROOT / "prompts.csv").open(newline="", encoding="utf-8") as file:
        prompts = {row["id"]: row for row in csv.DictReader(file)}
    data = json.loads((ROOT / "detections.json").read_text(encoding="utf-8"))
    pairs = {(row["id"], row["seed"]) for row in data["samples"]}
    assert len(data["samples"]) == len(pairs) == len(prompts) * len(data["seeds"])
    assert pairs == {(id_, seed) for id_ in prompts for seed in data["seeds"]}
    rows = summarize(prompts, data)
    with (ROOT / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    for threshold in THRESHOLDS:
        reference = data["reference"]
        hits = sum(score(dict(task="identity", food=r["food"]), r["detections"], threshold) for r in reference)
        print(f"Reference category presence at {threshold:.2f}: {hits}/{len(reference)}")


if __name__ == "__main__":
    main()
