import csv
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

DATA_FILE = "signbridge_data.csv"
MODEL_FILE = "signbridge_model.pkl"

if not os.path.exists(DATA_FILE):
    raise SystemExit("Data file not found!")

X = []
y = []

with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        try:
            features = [
                float(row[f"{axis}{i}"])
                for i in range(21)
                for axis in ("x", "y", "z")
            ]

            if len(features) == 63:
                X.append(features)
                y.append(row["label"])

        except (ValueError, KeyError):
            continue

if len(set(y)) < 2:
    raise SystemExit("Need at least two different labels!")

print(f"Total valid samples: {len(X)}")
print("Labels:", sorted(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n=== Initial Evaluation ===")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix (labels in this order):")
print(sorted(set(y)))
print(confusion_matrix(y_test, predictions, labels=sorted(set(y))))

joblib.dump(model, MODEL_FILE)

print(f"\nModel saved successfully: {MODEL_FILE}")
print("Next: test the model with new camera recordings.")