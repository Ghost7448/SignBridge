import csv
import os
from collections import Counter

FILE_NAME = "research_data.csv"

if not os.path.exists(FILE_NAME):
    print("ERROR: research_data.csv was not found!")
    raise SystemExit

with open(FILE_NAME, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

if not rows:
    print("ERROR: No samples found!")
    raise SystemExit

labels = Counter(row["label"] for row in rows)
positions = Counter(row["position"] for row in rows)

print(f"Total samples: {len(rows)}")

print("\nSamples per label:")
for label, count in sorted(labels.items()):
    print(f"{label}: {count}")

print("\nSamples per position:")
for position, count in sorted(positions.items()):
    print(f"{position}: {count}")

expected = {"CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM"}
missing = expected - set(positions)

if missing:
    print("\nMissing positions:", ", ".join(sorted(missing)))
else:
    print("\nAll five positions are present!")

bad_rows = 0

for row in rows:
    raw = [row.get(f"raw_{axis}{i}", "") for i in range(21) for axis in ("x", "y", "z")]
    relative = [row.get(f"rel_{axis}{i}", "") for i in range(21) for axis in ("x", "y", "z")]

    try:
        values = [float(v) for v in raw + relative]
        if len(values) != 126 or not all(map(__import__("math").isfinite, values)):
            bad_rows += 1
    except (ValueError, TypeError):
        bad_rows += 1

print(f"Rows with invalid coordinates: {bad_rows}")

if len(rows) == 1500 and not missing and bad_rows == 0:
    print("\nInitial data check passed!")
else:
    print("\nData needs review before training.")