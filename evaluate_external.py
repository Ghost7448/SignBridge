"""
Evaluate the trained models on NEW data that was never used for training
(a new person, new lighting, another day).

1) Record the new data into a separate file:
       python collect_research_data.py --out new_signer.csv
   (repeat for each sign / position; same label names as the training data)
2) Evaluate:
       python evaluate_external.py new_signer.csv

It loads model_raw.pkl and model_relative.pkl and prints accuracy, error rate,
a per-label breakdown and the confusion matrix. Results are saved to results/.
"""
import csv
import math
import os
import sys
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_COLS = [f"raw_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
REL_COLS = [f"rel_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
MODELS = {
    "RAW": ("model_raw.pkl", RAW_COLS),
    "RELATIVE": ("model_relative.pkl", REL_COLS),
}


def load(path):
    if not os.path.isfile(path):
        raise SystemExit(f"File not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    valid = []
    for row in rows:
        if not (row.get("label") or "").strip():
            continue
        try:
            values = [float(row[c]) for c in RAW_COLS + REL_COLS]
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in values):
            valid.append(row)
    if not valid:
        raise SystemExit("No valid rows found in " + path)
    return valid


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python evaluate_external.py <new_data.csv>")
    path = sys.argv[1]
    rows = load(path)
    y_true = [r["label"].strip() for r in rows]
    print(f"Loaded {len(rows)} samples from {path}")
    print("Samples per label:", dict(sorted(Counter(y_true).items())))
    if len({r.get("session_id", "") for r in rows}) < 3:
        print("Warning: very few recording sessions - collect more for a stronger test.")

    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)
    out_lines = [f"External test: {path} ({len(rows)} samples)"]
    for name, (model_file, cols) in MODELS.items():
        model_path = os.path.join(ROOT_DIR, model_file)
        if not os.path.isfile(model_path):
            print(f"\n{name}: {model_file} not found - skipped")
            continue
        model = joblib.load(model_path)
        X = np.asarray([[float(r[c]) for c in cols] for r in rows])
        start = time.perf_counter()
        pred = model.predict(X)
        ms = (time.perf_counter() - start) / len(X) * 1000
        acc = accuracy_score(y_true, pred) * 100
        labels = sorted(set(y_true) | set(pred))
        print(f"\n===== {name} =====")
        print(f"Accuracy: {acc:.2f}% | Error: {100 - acc:.2f}% | batch time {ms:.3f} ms/sample")
        print(classification_report(y_true, pred, zero_division=0))
        print("Confusion matrix (rows = true, columns = predicted):")
        print("labels:", labels)
        print(confusion_matrix(y_true, pred, labels=labels))
        out_lines.append(f"{name}: accuracy {acc:.2f}% | error {100 - acc:.2f}%")

    out_path = os.path.join(ROOT_DIR, "results", "external_test.txt")
    with open(out_path, "w", encoding="utf-8") as file:
        file.write("\n".join(out_lines) + "\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
