import csv
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

ROOT = Path("datasets/processed/hand_landmarks")
OUT = Path("models")
OUT.mkdir(parents=True, exist_ok=True)

FRAMES_PER_VIDEO = 12
POINTS = 21
COORDS = ("x", "y", "z")
FEATURES_PER_HAND = POINTS * len(COORDS)


def hand_features(row, hand_index):
    prefix = f"hand_{hand_index}_point_"

    points = []
    for i in range(POINTS):
        values = []
        for coord in COORDS:
            key = f"{prefix}{i}_{coord}"
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                return None
            if not np.isfinite(value):
                return None
            values.append(value)
        points.append(values)

    points = np.asarray(points, dtype=np.float32)

    # Normalize around the wrist and palm size.
    wrist = points[0].copy()
    points -= wrist

    scale = np.linalg.norm(points[9, :2])
    if scale < 1e-5:
        return None

    points /= scale
    return points.flatten()


def read_video(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    valid_rows = [
        row for row in rows
        if int(row.get("hand_count") or 0) > 0
    ]

    if not valid_rows:
        return None

    # Select frames spread across the detected portion of the video.
    indices = np.linspace(
        0, len(valid_rows) - 1,
        min(FRAMES_PER_VIDEO, len(valid_rows))
    ).astype(int)

    sequence = []

    for idx in indices:
        row = valid_rows[idx]
        features = []

        for hand_index in range(2):
            hand = hand_features(row, hand_index)
            if hand is None:
                features.extend([0.0] * FEATURES_PER_HAND)
            else:
                features.extend(hand.tolist())

        # Fixed size: 2 hands x 21 landmarks x 3 coordinates.
        sequence.extend(features)

    # Pad short videos to the same sequence length.
    frame_size = 2 * FEATURES_PER_HAND
    while len(sequence) < FRAMES_PER_VIDEO * frame_size:
        sequence.extend([0.0] * frame_size)

    return np.asarray(sequence, dtype=np.float32)


X, y, groups = [], [], []
skipped = 0

files = sorted(ROOT.rglob("*_landmarks.csv"))

for i, path in enumerate(files, 1):
    label = path.parent.name
    try:
        features = read_video(path)
        if features is None:
            skipped += 1
            continue

        X.append(features)
        y.append(label)
        groups.append(path.name)

    except Exception as exc:
        skipped += 1
        print(f"Skipped {path.name}: {exc}")

    if i % 50 == 0:
        print(f"Scanned {i}/{len(files)} videos")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)
groups = np.asarray(groups)

if len(X) == 0:
    raise SystemExit("No usable training videos found.")

labels, counts = np.unique(y, return_counts=True)
print("\nUsable videos:", len(X))
print("Skipped videos:", skipped)
print("Classes:", len(labels))
print("Minimum videos per class:", counts.min())

# Each row represents one whole video, so frames from the
# same video cannot leak into both train and test.
train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=3.0,
        class_weight="balanced",
        probability=True,
        random_state=42
    )
)

print("\nTraining model...")
model.fit(X[train_idx], y[train_idx])

predictions = model.predict(X[test_idx])
accuracy = accuracy_score(y[test_idx], predictions)

print("\n" + "=" * 50)
print("SIGNBRIDGE VIDEO-LEVEL EVALUATION")
print("=" * 50)
print("Training videos:", len(train_idx))
print("Test videos:", len(test_idx))
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(
    y[test_idx],
    predictions,
    zero_division=0
))

model_path = OUT / "signbridge_svm.pkl"

with model_path.open("wb") as f:
    pickle.dump({
        "model": model,
        "feature_size": X.shape[1],
        "frames_per_video": FRAMES_PER_VIDEO,
        "labels": labels.tolist(),
        "model_type": "video-level SVM baseline",
    }, f)

print("\nModel saved:", model_path)
print("Training complete.")