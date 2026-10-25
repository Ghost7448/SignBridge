import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_FILE = "research_data.csv"

RAW_COLS = [
    f"raw_{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

REL_COLS = [
    f"rel_{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

with open(DATA_FILE, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

positions = ["CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM"]
results = {"RAW": [], "RELATIVE": []}

for test_position in positions:
    train = [r for r in rows if r["position"] != test_position]
    test = [r for r in rows if r["position"] == test_position]

    print(f"\nTesting unseen position: {test_position}")

    y_train = [r["label"] for r in train]
    y_test = [r["label"] for r in test]

    for name, columns in [("RAW", RAW_COLS), ("RELATIVE", REL_COLS)]:
        X_train = np.array([[float(r[c]) for c in columns] for r in train])
        X_test = np.array([[float(r[c]) for c in columns] for r in test])

        model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100
        results[name].append(accuracy)

        print(f"{name}: {accuracy:.2f}%")

print("\n========== SUMMARY ==========")
print(f"Raw average:      {np.mean(results['RAW']):.2f}%")
print(f"Relative average: {np.mean(results['RELATIVE']):.2f}%")
print(
    "Difference:       "
    f"{np.mean(results['RELATIVE']) - np.mean(results['RAW']):+.2f} percentage points"
)
print("\nPreliminary results only; same signer and recording sessions.")