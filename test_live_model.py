import cv2
import mediapipe as mp
import joblib
import numpy as np
import math
import time

MODEL_FILE = "model_relative.pkl"

model = joblib.load(MODEL_FILE)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_relative(hand, width, height):
    points = [
        (p.x, p.y, p.z)
        for p in hand.landmark
    ]

    wrist = points[0]
    middle = points[9]

    wx, wy = wrist[0] * width, wrist[1] * height
    mx, my = middle[0] * width, middle[1] * height

    scale = math.hypot(mx - wx, my - wy)

    if scale < 1:
        return None

    features = []

    for x, y, z in points:
        features.extend([
            (x * width - wx) / scale,
            (y * height - wy) / scale,
            (z * width - wrist[2] * width) / scale
        ])

    return np.array(features).reshape(1, -1)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    raise SystemExit

print("Show one of your three experimental poses.")
print("Press Q to quit.")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:
    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            prediction_text = "Show your hand"

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                features = extract_relative(hand, width, height)

                if features is not None:
                    prediction = model.predict(features)[0]
                    prediction_text = f"Prediction: {prediction}"

            cv2.putText(
                frame,
                prediction_text,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("SignBridge - Live Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()