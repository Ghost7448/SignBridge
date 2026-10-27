import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path("datasets/processed/hand_landmarks")
OUT = Path("datasets/processed/training")
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(ROOT.rglob("*_landmarks.csv"))
report = []
usable = 0
empty = 0

for path in files:
    total = 0
    detected = 0
    label = path.parent.name

    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if int(row.get("hand_count") or 0) > 0:
                detected += 1

    rate = detected / total if total else 0
    status = "usable" if detected > 0 else "no_hands"

    if detected:
        usable += 1
    else:
        empty += 1

    report.append({
        "label": label,
        "video": path.name,
        "source_csv": str(path),
        "frames": total,
        "frames_with_hands": detected,
        "detection_rate": round(rate, 4),
        "status": status,
    })

out_file = OUT / "video_quality_report.csv"
with out_file.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=report[0].keys())
    writer.writeheader()
    writer.writerows(report)

print("Videos scanned:", len(files))
print("Videos with detected hands:", usable)
print("Videos with zero detected hands:", empty)
print("Report saved:", out_file)
print("No original data was modified.")