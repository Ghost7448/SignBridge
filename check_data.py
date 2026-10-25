import csv
import os
from collections import Counter

DATA_FILE = "signbridge_data.csv"

if not os.path.exists(DATA_FILE):
    print("ERROR: Data file not found!")
    raise SystemExit

with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

if not rows:
    print("ERROR: No data found!")
    raise SystemExit

counts = Counter(row["label"] for row in rows)

print("\n=== SignBridge Data Report ===")
print(f"Total samples: {len(rows)}")

for label, count in sorted(counts.items()):
    print(f"{label}: {count} samples")

expected = {"SIGN_1", "SIGN_2", "SIGN_3"}
missing = expected - set(counts)

if missing:
    print("\nStill missing:", ", ".join(sorted(missing)))
    print("Collect the missing poses before training.")
else:
    print("\nAll three labels are present!")

valid_rows = 0

for row in rows:
    try:
        values = [
            float(row[f"{axis}{i}"])
            for i in range(21)
            for axis in ("x", "y", "z")
        ]
        if len(values) == 63:
            valid_rows += 1
    except (ValueError, KeyError, TypeError):
        pass

print(f"Rows with valid coordinates: {valid_rows}")

if valid_rows == len(rows) and not missing:
    print("Data check passed!")
else:
    print("Data needs checking before training.")