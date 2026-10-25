import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import joblib
import numpy as np
import math
import threading

from PIL import Image, ImageTk

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


MODEL_FILE = "model_relative.pkl"

COLORS = {
    "bg": "#10141F",
    "panel": "#1B2231",
    "accent": "#D83B55",
    "text": "#F4F6FC",
    "muted": "#A8B1C5",
    "green": "#58D6A5",
}


class SignBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignBridge | Sign Recognition")
        self.root.geometry("1050x720")
        self.root.minsize(850, 600)
        self.root.configure(bg=COLORS["bg"])

        self.model = joblib.load(MODEL_FILE)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = None
        self.hands = None
        self.camera_running = False
        self.photo = None
        self.last_prediction = None
        self.last_spoken = None

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def build_ui(self):
        header = tk.Frame(self.root, bg=COLORS["bg"])
        header.pack(fill="x", padx=24, pady=(20, 12))

        tk.Label(
            header,
            text="SignBridge",
            font=("Segoe UI", 26, "bold"),
            bg=COLORS["bg"],
            fg=COLORS["text"],
        ).pack(side="left")

        tk.Label(
            header,
            text="HAND RECOGNITION • PROTOTYPE",
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["bg"],
            fg=COLORS["green"],
        ).pack(side="right", pady=12)

        tk.Label(
            self.root,
            text="تعرف على أوضاع اليد باستخدام الكاميرا",
            font=("Segoe UI", 16, "bold"),
            bg=COLORS["bg"],
            fg=COLORS["text"],
        ).pack(anchor="w", padx=26, pady=(0, 14))

        content = tk.Frame(self.root, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, padx=24, pady=(0, 18))

        camera_panel = tk.Frame(
            content, bg=COLORS["panel"], padx=12, pady=12
        )
        camera_panel.pack(side="left", fill="both", expand=True)

        tk.Label(
            camera_panel,
            text="LIVE CAMERA",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["panel"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(0, 10))

        self.camera_label = tk.Label(
            camera_panel,
            text="الكاميرا متوقفة\n\nاضغط تشغيل الكاميرا للبدء",
            font=("Segoe UI", 15),
            bg="#090D15",
            fg=COLORS["muted"],
            justify="center",
        )
        self.camera_label.pack(fill="both", expand=True)

        side = tk.Frame(
            content, bg=COLORS["panel"], padx=18, pady=18, width=270
        )
        side.pack(side="right", fill="y", padx=(14, 0))
        side.pack_propagate(False)

        tk.Label(
            side,
            text="نتيجة التعرّف",
            font=("Segoe UI", 15, "bold"),
            bg=COLORS["panel"],
            fg=COLORS["text"],
        ).pack(anchor="w")

        self.prediction_label = tk.Label(
            side,
            text="في انتظار الإشارة",
            font=("Segoe UI", 20, "bold"),
            bg=COLORS["panel"],
            fg=COLORS["green"],
            wraplength=230,
        )
        self.prediction_label.pack(anchor="w", pady=(22, 8))

        self.status_label = tk.Label(
            side,
            text="الحالة: جاهز",
            font=("Segoe UI", 10),
            bg=COLORS["panel"],
            fg=COLORS["muted"],
        )
        self.status_label.pack(anchor="w", pady=(0, 22))

        tk.Button(
            side,
            text="▶  تشغيل الكاميرا",
            command=self.start_camera,
            bg=COLORS["accent"],
            fg="white",
            activebackground="#B92D45",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
            pady=11,
        ).pack(fill="x", pady=5)

        tk.Button(
            side,
            text="■  إيقاف الكاميرا",
            command=self.stop_camera,
            bg="#30394C",
            fg="white",
            activebackground="#414D64",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 11),
            cursor="hand2",
            pady=10,
        ).pack(fill="x", pady=5)

        tk.Button(
            side,
            text="🔊  نطق النتيجة",
            command=self.speak_prediction,
            bg="#30394C",
            fg="white",
            activebackground="#414D64",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 11),
            cursor="hand2",
            pady=10,
        ).pack(fill="x", pady=5)

        tk.Label(
            side,
            text=(
                "ملاحظة بحثية:\n"
                "SIGN_1 وSIGN_2 وSIGN_3\n"
                "أوضاع تجريبية وليست ترجمات\n"
                "معتمدة للغة الإشارة المصرية."
            ),
            font=("Segoe UI", 9),
            justify="left",
            wraplength=230,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(25, 0))

        footer = tk.Label(
            self.root,
            text="SignBridge • Research Prototype",
            font=("Segoe UI", 9),
            bg=COLORS["bg"],
            fg=COLORS["muted"],
        )
        footer.pack(pady=(0, 10))

    @staticmethod
    def extract_relative(hand, width, height):
        points = [(p.x, p.y, p.z) for p in hand.landmark]

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
                (z * width - wrist[2] * width) / scale,
            ])

        return np.asarray(features, dtype=float).reshape(1, -1)

    def start_camera(self):
        if self.camera_running:
            return

        try:
            self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                messagebox.showerror(
                    "Camera error",
                    "تعذر فتح الكاميرا. اقفل أي برنامج بيستخدمها وجرب تاني."
                )
                return

            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )

            self.camera_running = True
            self.status_label.config(text="الحالة: الكاميرا تعمل")
            self.update_frame()

        except Exception as error:
            self.stop_camera()
            messagebox.showerror("Error", str(error))

    def update_frame(self):
        if not self.camera_running or self.cap is None:
            return

        try:
            success, frame = self.cap.read()

            if not success:
                self.status_label.config(text="تعذر قراءة الكاميرا")
                self.root.after(100, self.update_frame)
                return

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]

                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS
                )

                features = self.extract_relative(hand, width, height)

                if features is not None:
                    prediction = str(self.model.predict(features)[0])
                    self.last_prediction = prediction

                    cv2.putText(
                        frame,
                        prediction,
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (88, 214, 165),
                        2,
                    )

                    self.prediction_label.config(text=prediction)
            else:
                self.prediction_label.config(text="لم يتم رصد يد")

            display = cv2.resize(frame, (640, 420))
            image = Image.fromarray(display)
            self.photo = ImageTk.PhotoImage(image=image)
            self.camera_label.config(image=self.photo, text="")

        except Exception as error:
            self.status_label.config(text="حدث خطأ أثناء تشغيل الكاميرا")
            print("Camera error:", error)

        if self.camera_running:
            self.root.after(30, self.update_frame)

    def stop_camera(self):
        self.camera_running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.hands is not None:
            self.hands.close()
            self.hands = None

        self.camera_label.config(
            image="",
            text="الكاميرا متوقفة\n\nاضغط تشغيل الكاميرا للبدء",
        )
        self.photo = None
        self.status_label.config(text="الحالة: جاهز")

    def speak_prediction(self):
        if not self.last_prediction:
            messagebox.showinfo(
                "لا توجد نتيجة",
                "شغّل الكاميرا واعرض إحدى الإشارات التجريبية أولًا."
            )
            return

        if pyttsx3 is None:
            messagebox.showerror(
                "الصوت غير متاح",
                "افتح التيرمنال واكتب: pip install pyttsx3"
            )
            return

        # These are prototype labels, not real sign translations.
        spoken_text = {
            "SIGN_1": "الإشارة التجريبية واحد",
            "SIGN_2": "الإشارة التجريبية اثنان",
            "SIGN_3": "الإشارة التجريبية ثلاثة",
        }.get(self.last_prediction, "إشارة تجريبية")

        def speak():
            try:
                engine = pyttsx3.init()
                engine.say(spoken_text)
                engine.runAndWait()
                engine.stop()
            except Exception as error:
                print("Speech error:", error)

        threading.Thread(target=speak, daemon=True).start()

    def close_app(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignBridgeApp(root)
    root.mainloop()