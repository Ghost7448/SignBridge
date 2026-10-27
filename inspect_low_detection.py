import csv
from pathlib import Path

root = Path("datasets/processed/hand_landmarks")
target_label = "delwaty"

files = sorted((root / target_label).glob("*_landmarks.csv"))

if not files:
    raise SystemExit("No CSV files found for delwaty")

for path in files:
    total = 0
    detected = 0

    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if int(row["hand_count"] or 0) > 0:
                detected += 1

    rate = 100 * detected / total if total else 0
    print(f"{path.name}: {detected}/{total} frames ({rate:.1f}%)")