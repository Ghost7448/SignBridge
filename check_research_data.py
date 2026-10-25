import csv
import os
import math
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = os.path.join(ROOT_DIR, "research_data.csv")
EXPECTED_POSITIONS = {"CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM"}
RAW_COLUMNS = [f"raw_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
REL_COLUMNS = [f"rel_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def main():
    if not os.path.isfile(FILE_NAME):
        print(f"ERROR: research_data.csv was not found at {FILE_NAME}")
        raise SystemExit(1)

    with open(FILE_NAME, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        print("ERROR: No samples found!")
        raise SystemExit(1)

    labels = Counter((row.get("label") or "").strip() for row in rows)
    positions = Counter((row.get("position") or "").strip().upper() for row in rows)

    print(f"Total samples: {len(rows)}")
    print("\nSamples per label:")
    for label, count in sorted(labels.items()):
        print(f"{label or '(empty label)'}: {count}")

    print("\nSamples per position:")
    for position, count in sorted(positions.items()):
        print(f"{position or '(empty position)'}: {count}")

    missing = EXPECTED_POSITIONS - set(positions)
    if missing:
        print("\nMissing positions:", ", ".join(sorted(missing)))
    else:
        print("\nAll five positions are present!")

    bad_rows = 0
    for row in rows:
        try:
            values = [float(row[column]) for column in RAW_COLUMNS + REL_COLUMNS]
            if len(values) != 126 or not all(math.isfinite(value) for value in values):
                bad_rows += 1
        except (KeyError, TypeError, ValueError):
            bad_rows += 1

    empty_labels = sum(1 for row in rows if not (row.get("label") or "").strip())
    print(f"Rows with invalid coordinates: {bad_rows}")
    print(f"Rows with empty labels: {empty_labels}")

    if not missing and bad_rows == 0 and empty_labels == 0:
        print("\nData integrity check passed.")
    else:
        print("\nData needs review before training.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
