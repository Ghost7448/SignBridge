"""Inspect an extracted Mendeley Egyptian Sign Language video dataset.

Expected location: datasets/raw/ (extract the downloaded ZIP there).
This script only inventories files; it does not train or replace the live model.
"""
from collections import Counter
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "datasets" / "raw"
OUT_CSV = ROOT / "datasets" / "esl_video_inventory.csv"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}

def main():
    if not DATA_ROOT.exists():
        print("Dataset folder not found:", DATA_ROOT)
        print("Download Mendeley ESL Version 2 and extract its ZIP into datasets/raw/")
        print("Guide: datasets/README.md")
        return 1

    videos = sorted(p for p in DATA_ROOT.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print("No video files found under:", DATA_ROOT)
        print("Check that the ZIP was extracted and contains video files.")
        return 1

    # Folder directly containing each video is used as a tentative class label.
    # Dataset layouts can differ, so verify these labels before training.
    rows = []
    counts = Counter()
    for path in videos:
        label = path.parent.name
        counts[label] += 1
        rows.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "tentative_label": label,
            "extension": path.suffix.lower(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "relative_path", "tentative_label", "extension", "size_mb"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Videos found: {len(videos)}")
    print(f"Tentative folder labels: {len(counts)}")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")
    print(f"Inventory saved to: {OUT_CSV}")
    print("\nNote: folder names are only tentative labels. Verify the dataset layout and labels before training.")
    print("This script does not train or change the SignBridge live model.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
