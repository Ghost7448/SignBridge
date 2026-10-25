import csv
import os
import math
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, "research_data.csv")
MODEL_FILE = os.path.join(ROOT_DIR, "model_relative.pkl")
REL_COLS = [f"rel_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def main():
    if not os.path.isfile(DATA_FILE):
        raise SystemExit(f"Research data file not found: {DATA_FILE}")

    X, y = [], []
    with open(DATA_FILE, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        missing = [name for name in ["label"] + REL_COLS if name not in fields]
        if missing:
            raise SystemExit("CSV is missing required columns: " + ", ".join(missing[:8]))
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                continue
            try:
                values = [float(row[column]) for column in REL_COLS]
                if len(values) == 63 and all(math.isfinite(value) for value in values):
                    X.append(values)
                    y.append(label)
            except (ValueError, TypeError):
                continue

    counts = Counter(y)
    if len(counts) < 2:
        raise SystemExit("At least two labels with valid samples are required.")
    if min(counts.values()) < 2:
        raise SystemExit("Each label needs at least two valid samples for a stratified split.")

    print(f"Valid samples: {len(X)}")
    print("Samples per label:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    )
    print("\nTraining relative-coordinate model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\nRandom split accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(classification_report(y_test, predictions, zero_division=0))
    joblib.dump(model, MODEL_FILE)
    print(f"Saved model: {MODEL_FILE}")
    print("Note: a random split may overestimate performance when samples come from the same recording sessions.")


if __name__ == "__main__":
    main()
