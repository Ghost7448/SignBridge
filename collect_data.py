import cv2
import mediapipe as mp
import csv
import os

DATA_FILE = "signbridge_data.csv"
SAMPLES = 200

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

labels = {
    "1": "SIGN_1",
    "2": "SIGN_2",
    "3": "SIGN_3"
}

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        print("\nChoose a hand pose to collect:")
        print("1 - SIGN_1 (first pose)")
        print("2 - SIGN_2 (second pose)")
        print("3 - SIGN_3 (third pose)")
        print("Q - Quit")

        choice = input("Your choice: ").strip().lower()

        if choice == "q":
            break

        if choice not in labels:
            print("Invalid choice. Try again.")
            continue

        label = labels[choice]
        count = 0

        file_exists = os.path.exists(DATA_FILE)
        write_header = (
            not file_exists or os.path.getsize(DATA_FILE) == 0
        )

        with open(
            DATA_FILE, "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            if write_header:
                header = ["label"]
                for i in range(21):
                    header.extend([f"x{i}", f"y{i}", f"z{i}"])
                writer.writerow(header)

            print(f"\nShow pose {label} to the camera.")
            print("Press S in the camera window to start.")
            print("Press Q in the camera window to cancel.")

            started = False

            while cap.isOpened() and count < SAMPLES:
                success, frame = cap.read()

                if not success:
                    print("Could not read camera.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                    if started:
                        row = [label]
                        for point in hand.landmark:
                            row.extend([point.x, point.y, point.z])

                        writer.writerow(row)
                        file.flush()
                        count += 1

                message = (
                    f"{label}: {count}/{SAMPLES}"
                    if started else "Press S to start | Q to cancel"
                )

                cv2.putText(
                    frame, message, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0), 2
                )

                cv2.imshow("SignBridge - Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("s") and not started:
                    started = True

                if key == ord("q"):
                    break

        print(f"Saved {count} samples for {label}.")

cap.release()
cv2.destroyAllWindows()

print(f"\nData file: {os.path.abspath(DATA_FILE)}")
print("Collection finished.")