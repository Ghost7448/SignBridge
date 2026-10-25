import cv2
import mediapipe as mp
import joblib

MODEL_FILE = "signbridge_model.pkl"

model = joblib.load(MODEL_FILE)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Could not open camera!")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Could not read camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label = "No hand detected"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            features = [
                coordinate
                for point in hand.landmark
                for coordinate in (point.x, point.y, point.z)
            ]

            label = model.predict([features])[0]

        cv2.rectangle(frame, (0, 0), (440, 55), (30, 30, 30), -1)

        cv2.putText(
            frame, f"Prediction: {label}",
            (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.imshow("SignBridge - Live Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()