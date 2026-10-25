import csv
import os
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, "research_data.csv")
POSITIONS = ["CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM"]

RAW_COLS = [f"raw_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
REL_COLS = [f"rel_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def load_rows():
    if not os.path.isfile(DATA_FILE):
        raise SystemExit(f"Data file not found: {DATA_FILE}")
    with open(DATA_FILE, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("research_data.csv has no samples.")

    valid = []
    for row in rows:
        if not row.get("label") or not row.get("position"):
            continue
        try:
            values = [float(row[c]) for c in RAW_COLS + REL_COLS]
            if len(values) == 126 and all(math.isfinite(v) for v in values):
                valid.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    if not valid:
        raise SystemExit("No valid rows with complete raw_ and rel_ features were found.")
    return valid


def evaluate(rows, test_position, columns):
    train = [row for row in rows if row["position"].upper() != test_position]
    test = [row for row in rows if row["position"].upper() == test_position]
    if not train or not test:
        return None, len(train), len(test)
    y_train = [row["label"] for row in train]
    y_test = [row["label"] for row in test]
    if len(set(y_train)) < 2 or not set(y_test).issubset(set(y_train)):
        return None, len(train), len(test)

    try:
        X_train = np.asarray([[float(row[c]) for c in columns] for row in train])
        X_test = np.asarray([[float(row[c]) for c in columns] for row in test])
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        )
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test)) * 100, len(train), len(test)
    except ValueError as error:
        print(f"Skipped {test_position}: {error}")
        return None, len(train), len(test)


def main():
    rows = load_rows()
    print(f"Valid samples loaded: {len(rows)}")
    print("Evaluation: leave one camera position out")
    results = {"RAW": [], "RELATIVE": []}

    for position in POSITIONS:
        if not any(r["position"].upper() == position for r in rows):
            print(f"\n{position}: skipped (no test samples)")
            continue
        raw, _, test_count = evaluate(rows, position, RAW_COLS)
        relative, _, _ = evaluate(rows, position, REL_COLS)
        print(f"\nHeld-out position: {position} | test samples: {test_count}")
        print(f"RAW:      {raw:.2f}%" if raw is not None else "RAW:      unavailable")
        print(f"RELATIVE: {relative:.2f}%" if relative is not None else "RELATIVE: unavailable")
        if raw is not None:
            results["RAW"].append(raw)
        if relative is not None:
            results["RELATIVE"].append(relative)

    print("\n========== SUMMARY ==========")
    raw_avg = float(np.mean(results["RAW"])) if results["RAW"] else None
    rel_avg = float(np.mean(results["RELATIVE"])) if results["RELATIVE"] else None
    print(f"Raw average:      {raw_avg:.2f}%" if raw_avg is not None else "Raw average:      unavailable")
    print(f"Relative average: {rel_avg:.2f}%" if rel_avg is not None else "Relative average: unavailable")
    if raw_avg is not None and rel_avg is not None:
        print(f"Difference:       {rel_avg - raw_avg:+.2f} percentage points")
    print("\nResearch caution: these results are preliminary and may not generalize to new signers.")


if __name__ == "__main__":
    main()
