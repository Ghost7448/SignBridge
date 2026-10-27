import csv
from pathlib import Path
from collections import defaultdict

root = Path("datasets/processed/hand_landmarks")
files = list(root.rglob("*_landmarks.csv"))

stats = defaultdict(lambda: {
    "videos": 0,
    "frames": 0,
    "detected": 0,
})

for path in files:
    label = path.parent.name
    video_frames = 0
    video_detected = 0

    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            video_frames += 1
            if int(row["hand_count"] or 0) > 0:
                video_detected += 1

    stats[label]["videos"] += 1
    stats[label]["frames"] += video_frames
    stats[label]["detected"] += video_detected

print("=== LANDMARK QUALITY REPORT ===")
print("Videos:", len(files))
print()

for label, s in sorted(stats.items()):
    frames = s["frames"]
    detected = s["detected"]
    pct = 100 * detected / frames if frames else 0

    print(
        f"{label}: videos={s['videos']}, "
        f"frames={frames}, hands_detected={detected}, "
        f"rate={pct:.1f}%"
    )

print("\nReport complete.")