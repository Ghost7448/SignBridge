import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp


# =========================
# Configuration
# =========================

DATASET_DIR = Path("datasets/raw/dataset - rebirth")
OUTPUT_DIR = Path("datasets/processed/hand_landmarks")
SUMMARY_FILE = OUTPUT_DIR / "extraction_summary.csv"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

mp_hands = mp.solutions.hands


def create_columns():
    columns = [
        "label",
        "video_name",
        "video_relative_path",
        "frame_index",
        "timestamp_ms",
        "fps",
        "frame_width",
        "frame_height",
        "hand_count",
    ]

    for hand_index in range(MAX_HANDS):
        columns.append(f"hand_{hand_index}_label")

        for point_index in range(21):
            prefix = f"hand_{hand_index}_point_{point_index}"
            columns.extend([
                f"{prefix}_x",
                f"{prefix}_y",
                f"{prefix}_z",
            ])

    return columns


COLUMNS = create_columns()


def discover_videos():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR.resolve()}"
        )

    return sorted(
        (
            path for path in DATASET_DIR.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def empty_row(label, video, relative_path, frame_index,
              timestamp_ms, fps, width, height):
    row = {
        "label": label,
        "video_name": video.name,
        "video_relative_path": str(relative_path),
        "frame_index": frame_index,
        "timestamp_ms": round(timestamp_ms, 3),
        "fps": round(fps, 3),
        "frame_width": width,
        "frame_height": height,
        "hand_count": 0,
    }

    for hand_index in range(MAX_HANDS):
        row[f"hand_{hand_index}_label"] = ""

        for point_index in range(21):
            prefix = f"hand_{hand_index}_point_{point_index}"
            row[f"{prefix}_x"] = ""
            row[f"{prefix}_y"] = ""
            row[f"{prefix}_z"] = ""

    return row


def extract_video(video, overwrite=False):
    relative_path = video.relative_to(DATASET_DIR)
    label = relative_path.parent.name

    # Keep each label in its own output directory.
    output_folder = OUTPUT_DIR / label
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / f"{video.stem}_landmarks.csv"
    temp_file = output_folder / f"{video.stem}_landmarks.tmp"

    if output_file.exists() and not overwrite:
        return {
            "video": str(relative_path),
            "label": label,
            "status": "skipped_existing",
            "frames": 0,
            "frames_with_hands": 0,
            "error": "",
        }

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        cap.release()
        return {
            "video": str(relative_path),
            "label": label,
            "status": "error",
            "frames": 0,
            "frames_with_hands": 0,
            "error": "OpenCV could not open video",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 0.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_index = 0
    frames_with_hands = 0
    started = time.time()

    try:
        with temp_file.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_HANDS,
                model_complexity=0,
                min_detection_confidence=DETECTION_CONFIDENCE,
                min_tracking_confidence=TRACKING_CONFIDENCE,
            ) as detector:

                while True:
                    ok, frame = cap.read()

                    if not ok:
                        break

                    timestamp_ms = (
                        frame_index * 1000.0 / fps if fps > 0 else 0.0
                    )

                    row = empty_row(
                        label=label,
                        video=video,
                        relative_path=relative_path,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        fps=fps,
                        width=width,
                        height=height,
                    )

                    # MediaPipe requires RGB input.
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = detector.process(rgb)

                    landmarks = result.multi_hand_landmarks or []
                    handedness = result.multi_handedness or []

                    row["hand_count"] = len(landmarks)

                    if landmarks:
                        frames_with_hands += 1

                    for hand_index, hand in enumerate(
                        landmarks[:MAX_HANDS]
                    ):
                        if hand_index < len(handedness):
                            classification = (
                                handedness[hand_index].classification
                            )
                            if classification:
                                row[f"hand_{hand_index}_label"] = (
                                    classification[0].label
                                )

                        for point_index, point in enumerate(
                            hand.landmark
                        ):
                            prefix = (
                                f"hand_{hand_index}_point_{point_index}"
                            )
                            row[f"{prefix}_x"] = round(point.x, 7)
                            row[f"{prefix}_y"] = round(point.y, 7)
                            row[f"{prefix}_z"] = round(point.z, 7)

                    writer.writerow(row)
                    frame_index += 1

                    if frame_index % 100 == 0:
                        file.flush()

        cap.release()

        if frame_index == 0:
            temp_file.unlink(missing_ok=True)
            return {
                "video": str(relative_path),
                "label": label,
                "status": "error",
                "frames": 0,
                "frames_with_hands": 0,
                "error": "Video contained no readable frames",
            }

        # Only publish the final CSV after successful completion.
        os.replace(temp_file, output_file)

        return {
            "video": str(relative_path),
            "label": label,
            "status": "success",
            "frames": frame_index,
            "frames_with_hands": frames_with_hands,
            "error": "",
        }

    except Exception as exc:
        cap.release()
        temp_file.unlink(missing_ok=True)

        return {
            "video": str(relative_path),
            "label": label,
            "status": "error",
            "frames": frame_index,
            "frames_with_hands": frames_with_hands,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from videos."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess videos whose output CSV already exists.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        videos = discover_videos()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    if not videos:
        print("No videos found. Check the dataset directory.")
        return 1

    print("=" * 60)
    print("SignBridge - Hand Landmark Extraction")
    print("=" * 60)
    print(f"Dataset: {DATASET_DIR.resolve()}")
    print(f"Videos found: {len(videos)}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print("Processing one video at a time...")
    print()

    results = []

    for index, video in enumerate(videos, start=1):
        print(
            f"[{index}/{len(videos)}] "
            f"{video.parent.name}/{video.name}",
            flush=True,
        )

        result = extract_video(video, overwrite=args.overwrite)
        results.append(result)

        print(
            f"  Status: {result['status']} | "
            f"Frames: {result['frames']} | "
            f"Frames with hands: {result['frames_with_hands']}"
            + (f" | Error: {result['error']}" if result["error"] else ""),
            flush=True,
        )

        # Save progress after every video.
        with SUMMARY_FILE.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            fields = [
                "video",
                "label",
                "status",
                "frames",
                "frames_with_hands",
                "error",
            ]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

    successful = sum(r["status"] == "success" for r in results)
    skipped = sum(r["status"] == "skipped_existing" for r in results)
    failed = sum(r["status"] == "error" for r in results)
    total_frames = sum(r["frames"] for r in results)
    detected_frames = sum(r["frames_with_hands"] for r in results)

    print()
    print("=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Videos processed: {successful}")
    print(f"Existing outputs skipped: {skipped}")
    print(f"Failed videos: {failed}")
    print(f"Frames processed: {total_frames}")
    print(f"Frames with at least one detected hand: {detected_frames}")
    print(f"Summary file: {SUMMARY_FILE.resolve()}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())