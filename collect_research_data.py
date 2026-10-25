import cv2
import mediapipe as mp
import csv
import os
import math
import uuid

DATA_FILE = "research_data.csv"
SAMPLES_TO_COLLECT = 100
FRAME_INTERVAL = 3

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_features(hand_landmarks, frame_width, frame_height):
    points = []

    for landmark in hand_landmarks.landmark:
        points.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    # Model A: original MediaPipe coordinates
    raw_features = []
    for point in points:
        raw_features.extend(point)

    # Model B: wrist-relative coordinates, normalized by hand size
    wrist = points[0]
    middle_mcp = points[9]

    wrist_x = wrist[0] * frame_width
    wrist_y = wrist[1] * frame_height
    middle_x = middle_mcp[0] * frame_width
    middle_y = middle_mcp[1] * frame_height

    scale = math.hypot(
        middle_x - wrist_x,
        middle_y - wrist_y
    )

    if scale < 1:
        return None, None

    relative_features = []

    for x, y, z in points:
        relative_x = (x * frame_width - wrist_x) / scale
        relative_y = (y * frame_height - wrist_y) / scale
        relative_z = (z * frame_width - wrist[2] * frame_width) / scale

        relative_features.extend([
            relative_x,
            relative_y,
            relative_z
        ])

    return raw_features, relative_features


def main():
    label = input(
        "Enter the sign label (example: SIGN_1): "
    ).strip().upper()

    if not label:
        print("Label cannot be empty.")
        return

    print("\nChoose the hand position in the camera:")
    print("1 - CENTER")
    print("2 - LEFT")
    print("3 - RIGHT")
    print("4 - TOP")
    print("5 - BOTTOM")

    position_choices = {
        "1": "CENTER",
        "2": "LEFT",
        "3": "RIGHT",
        "4": "TOP",
        "5": "BOTTOM"
    }

    choice = input("Position number: ").strip()

    if choice not in position_choices:
        print("Invalid position.")
        return

    position = position_choices[choice]
    session_id = str(uuid.uuid4())[:8]

    raw_headers = [
        f"raw_{axis}{i}"
        for i in range(21)
        for axis in ("x", "y", "z")
    ]

    relative_headers = [
        f"rel_{axis}{i}"
        for i in range(21)
        for axis in ("x", "y", "z")
    ]

    headers = [
        "label",
        "position",
        "session_id",
        "sample_number"
    ] + raw_headers + relative_headers

    file_exists = os.path.exists(DATA_FILE)
    file_is_empty = (
        not file_exists or os.path.getsize(DATA_FILE) == 0
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open the camera.")
        return

    collected = 0
    frame_count = 0
    recording = False

    print("\nKeep the SAME hand sign throughout this collection.")
    print("Move your hand to the selected camera position.")
    print("Press S to start, Q to quit.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            with open(
                DATA_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                if file_is_empty:
                    writer.writerow(headers)

                while cap.isOpened():
                    success, frame = cap.read()

                    if not success:
                        print("Camera frame could not be read.")
                        break

                    frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]

                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]

                        mp_draw.draw_landmarks(
                            frame,
                            hand,
                            mp_hands.HAND_CONNECTIONS
                        )

                        if recording:
                            frame_count += 1

                            if frame_count % FRAME_INTERVAL == 0:
                                raw, relative = get_features(
                                    hand,
                                    width,
                                    height
                                )

                                if raw is not None:
                                    collected += 1

                                    writer.writerow([
                                        label,
                                        position,
                                        session_id,
                                        collected
                                    ] + raw + relative)

                                    file.flush()

                    status = (
                        f"{label} | {position} | "
                        f"Samples: {collected}/{SAMPLES_TO_COLLECT}"
                    )

                    cv2.putText(
                        frame,
                        status,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                    if not recording:
                        cv2.putText(
                            frame,
                            "Press S to start",
                            (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

                    cv2.imshow(
                        "SignBridge Research Data Collector",
                        frame
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("s") and not recording:
                        recording = True
                        print("Recording started...")

                    elif key == ord("q"):
                        break

                    if collected >= SAMPLES_TO_COLLECT:
                        print("Collection complete!")
                        break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nCollected {collected} samples.")
    print(f"Saved to: {DATA_FILE}")
    print(f"Label: {label} | Position: {position}")


if __name__ == "__main__":
    main()