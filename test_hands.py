import cv2
import mediapipe as mp
from pathlib import Path

root = Path("datasets/raw/dataset - rebirth")
video = next(root.rglob("delwaty_1297.mp4"))

cap = cv2.VideoCapture(str(video))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
samples = min(20, total)

detected = 0
checked = 0

with mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.3,
) as hands:
    for i in range(samples):
        index = int(i * (total - 1) / max(samples - 1, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()

        if not ok:
            continue

        # Enlarge frame 2x before detection.
        frame = cv2.resize(
            frame, None, fx=2, fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        result = hands.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        count = len(result.multi_hand_landmarks or [])
        detected += count > 0
        checked += 1
        print(f"Frame {index}: {count} hand(s)")

cap.release()

rate = 100 * detected / max(checked, 1)
print(f"\nDetected hands: {detected}/{checked}")
print(f"Detection rate: {rate:.1f}%")