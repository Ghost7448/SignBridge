import os
import threading
import tkinter as tk
from tkinter import messagebox
import math

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, "model_relative.pkl")

COLORS = {
    "bg": "#10141F",
    "panel": "#1B2231",
    "accent": "#D83B55",
    "text": "#F4F6FC",
    "muted": "#A8B1C5",
    "green": "#58D6A5",
    "warning": "#F5C26B",
}


def extract_relative(hand, width, height):
    """Create the same 63 relative features used by the research collector."""
    points = [(p.x, p.y, p.z) for p in hand.landmark]
    wrist = points[0]
    middle = points[9]

    wx, wy = wrist[0] * width, wrist[1] * height
    mx, my = middle[0] * width, middle[1] * height
    scale = math.hypot(mx - wx, my - wy)

    if not math.isfinite(scale) or scale < 1:
        return None

    features = []
    for x, y, z in points:
        features.extend([
            (x * width - wx) / scale,
            (y * height - wy) / scale,
            (z * width - wrist[2] * width) / scale,
        ])

    values = np.asarray(features, dtype=np.float32).reshape(1, -1)
    if values.shape != (1, 63) or not np.isfinite(values).all():
        return None
    return values


class SignBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignBridge | التعرف على أوضاع اليد")
        self.root.geometry("1080x740")
        self.root.minsize(850, 620)
        self.root.configure(bg=COLORS["bg"])

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.model = None
        self.cap = None
        self.hands = None
        self.camera_running = False
        self.photo = None
        self.last_prediction = None
        self.last_confidence = None
        self.speech_lock = threading.Lock()

        self.build_ui()
        self.load_model()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def build_ui(self):
        header = tk.Frame(self.root, bg=COLORS["bg"])
        header.pack(fill="x", padx=24, pady=(18, 10))

        tk.Label(
            header, text="SignBridge", font=("Segoe UI", 26, "bold"),
            bg=COLORS["bg"], fg=COLORS["text"]
        ).pack(side="left")

        tk.Label(
            header, text="HAND POSE RECOGNITION • RESEARCH PROTOTYPE",
            font=("Segoe UI", 9, "bold"), bg=COLORS["bg"], fg=COLORS["green"]
        ).pack(side="right", pady=12)

        tk.Label(
            self.root, text="تعرّف على أوضاع اليد باستخدام الكاميرا",
            font=("Segoe UI", 16, "bold"), bg=COLORS["bg"], fg=COLORS["text"]
        ).pack(anchor="w", padx=26, pady=(0, 14))

        content = tk.Frame(self.root, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        camera_panel = tk.Frame(content, bg=COLORS["panel"], padx=12, pady=12)
        camera_panel.pack(side="left", fill="both", expand=True)

        tk.Label(
            camera_panel, text="LIVE CAMERA", font=("Segoe UI", 10, "bold"),
            bg=COLORS["panel"], fg=COLORS["muted"]
        ).pack(anchor="w", pady=(0, 10))

        self.camera_label = tk.Label(
            camera_panel, text="الكاميرا متوقفة\n\nاضغط «تشغيل الكاميرا» للبدء",
            font=("Segoe UI", 14), bg="#090D15", fg=COLORS["muted"], justify="center"
        )
        self.camera_label.pack(fill="both", expand=True)

        side = tk.Frame(content, bg=COLORS["panel"], padx=18, pady=18, width=280)
        side.pack(side="right", fill="y", padx=(14, 0))
        side.pack_propagate(False)

        tk.Label(
            side, text="نتيجة التعرّف", font=("Segoe UI", 14, "bold"),
            bg=COLORS["panel"], fg=COLORS["text"]
        ).pack(anchor="w")

        self.prediction_label = tk.Label(
            side, text="في انتظار الإشارة", font=("Segoe UI", 19, "bold"),
            bg=COLORS["panel"], fg=COLORS["green"], wraplength=235, justify="left"
        )
        self.prediction_label.pack(anchor="w", pady=(18, 5))

        self.confidence_label = tk.Label(
            side, text="درجة الثقة: —", font=("Segoe UI", 10),
            bg=COLORS["panel"], fg=COLORS["muted"]
        )
        self.confidence_label.pack(anchor="w", pady=(0, 16))

        self.status_label = tk.Label(
            side, text="الحالة: جارٍ تجهيز النظام", font=("Segoe UI", 10),
            bg=COLORS["panel"], fg=COLORS["muted"], wraplength=235, justify="left"
        )
        self.status_label.pack(anchor="w", pady=(0, 18))

        self.make_button(side, "▶  تشغيل الكاميرا", self.start_camera, COLORS["accent"])
        self.make_button(side, "■  إيقاف الكاميرا", self.stop_camera, "#30394C")
        self.make_button(side, "🔊  نطق النتيجة", self.speak_prediction, "#30394C")
        self.make_button(side, "ℹ  معلومات المشروع", self.show_about, "#30394C")

        tk.Label(
            side,
            text="ملاحظة بحثية:\nSIGN_1 / SIGN_2 / SIGN_3\nأوضاع تجريبية وليست ترجمات معتمدة للغة الإشارة المصرية.",
            font=("Segoe UI", 9), justify="left", wraplength=235,
            bg=COLORS["panel"], fg=COLORS["warning"]
        ).pack(anchor="w", pady=(20, 0))

        tk.Label(
            self.root,
            text="SignBridge • النتائج أولية وتعتمد على بيانات التدريب المتاحة",
            font=("Segoe UI", 9), bg=COLORS["bg"], fg=COLORS["muted"]
        ).pack(pady=(0, 10))

    @staticmethod
    def make_button(parent, text, command, color):
        tk.Button(
            parent, text=text, command=command, bg=color, fg="white",
            activebackground=color, activeforeground="white", relief="flat",
            font=("Segoe UI", 10, "bold"), cursor="hand2", pady=10
        ).pack(fill="x", pady=4)

    def load_model(self):
        if not os.path.isfile(MODEL_FILE):
            self.status_label.config(
                text="النموذج غير موجود. ضع model_relative.pkl بجوار app.py أو أعد تدريب النموذج."
            )
            return
        try:
            self.model = joblib.load(MODEL_FILE)
            if not hasattr(self.model, "predict"):
                raise TypeError("ملف النموذج لا يحتوي على مصنف صالح.")
            self.status_label.config(text="الحالة: النموذج جاهز")
        except Exception as error:
            self.model = None
            self.status_label.config(text=f"تعذر تحميل النموذج: {error}")

    def start_camera(self):
        if self.camera_running:
            return
        if self.model is None:
            messagebox.showerror(
                "النموذج غير جاهز",
                "لم يتم تحميل model_relative.pkl بنجاح.\nراجع README.md أو أعد تدريب النموذج."
            )
            return
        try:
            # Use a low capture resolution to reduce CPU load on older laptops.
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 24)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                messagebox.showerror(
                    "مشكلة في الكاميرا",
                    "تعذر فتح الكاميرا. أغلق أي برنامج يستخدمها ثم حاول مرة أخرى."
                )
                return

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            self.camera_running = True
            self.status_label.config(text="الحالة: الكاميرا تعمل")
            self.update_frame()
        except Exception as error:
            self.stop_camera()
            messagebox.showerror("خطأ في التشغيل", str(error))

    def update_frame(self):
        if not self.camera_running or self.cap is None:
            return
        try:
            success, frame = self.cap.read()
            if not success:
                self.status_label.config(text="تعذر قراءة إطار من الكاميرا")
                self.root.after(80, self.update_frame)
                return

            frame = cv2.flip(frame, 1)

            # Run MediaPipe on a smaller frame; this is much lighter on the CPU.
            small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Skip alternate frames to keep the interface responsive.
            if not hasattr(self, "_frame_counter"):
                self._frame_counter = 0
            self._frame_counter += 1
            if self._frame_counter % 2 == 0:
                result = self.hands.process(rgb)

                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    self.mp_draw.draw_landmarks(
                        small, hand, self.mp_hands.HAND_CONNECTIONS
                    )
                    features = extract_relative(hand, 320, 240)

                    if features is not None:
                        prediction = str(self.model.predict(features)[0])
                        confidence = None
                        if hasattr(self.model, "predict_proba"):
                            try:
                                confidence = float(
                                    np.max(self.model.predict_proba(features)[0])
                                )
                            except Exception:
                                confidence = None

                        self.last_prediction = prediction
                        self.last_confidence = confidence
                        self.prediction_label.config(text=prediction)
                        self.confidence_label.config(
                            text=f"درجة ثقة النموذج: {confidence * 100:.1f}%"
                            if confidence is not None else "درجة الثقة: غير متاحة"
                        )
                        cv2.putText(
                            small, prediction, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (88, 214, 165), 2
                        )
                else:
                    self.prediction_label.config(text="لم يتم رصد يد")
                    self.confidence_label.config(text="درجة الثقة: —")
                    self.last_prediction = None
                    self.last_confidence = None

            # Display a modest preview size to reduce UI rendering overhead.
            display = cv2.resize(small, (480, 360), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=image)
            self.camera_label.config(image=self.photo, text="")
        except Exception as error:
            self.status_label.config(text=f"حدث خطأ أثناء الكاميرا: {error}")

        if self.camera_running:
            self.root.after(15, self.update_frame)

    def stop_camera(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        if hasattr(self, "camera_label"):
            self.camera_label.config(
                image="", text="الكاميرا متوقفة\n\nاضغط «تشغيل الكاميرا» للبدء"
            )
            self.photo = None
        if hasattr(self, "status_label") and self.model is not None:
            self.status_label.config(text="الحالة: النموذج جاهز")

    def speak_prediction(self):
        if not self.last_prediction:
            messagebox.showinfo("لا توجد نتيجة", "شغّل الكاميرا واعرض إحدى الأوضاع التجريبية أولًا.")
            return
        if pyttsx3 is None:
            messagebox.showerror(
                "الصوت غير متاح",
                "ثبّت مكتبة الصوت من داخل البيئة الافتراضية بالأمر: pip install pyttsx3"
            )
            return

        spoken_text = {
            "SIGN_1": "الوضع التجريبي واحد",
            "SIGN_2": "الوضع التجريبي اثنان",
            "SIGN_3": "الوضع التجريبي ثلاثة",
        }.get(self.last_prediction, f"الوضع التجريبي {self.last_prediction}")

        def speak():
            try:
                with self.speech_lock:
                    engine = pyttsx3.init()
                    engine.say(spoken_text)
                    engine.runAndWait()
                    engine.stop()
            except Exception as error:
                print("Speech error:", error)

        threading.Thread(target=speak, daemon=True).start()

    def show_about(self):
        messagebox.showinfo(
            "عن SignBridge",
            "SignBridge مشروع بحثي تعليمي يستخدم MediaPipe لاستخراج معالم اليد "
            "ومصنفًا مدربًا للتعرف على أوضاع تجريبية.\n\n"
            "الفئات الحالية ليست ترجمة موثقة للغة الإشارة المصرية. "
            "درجة ثقة النموذج ليست ضمانًا لصحة التنبؤ."
        )

    def close_app(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignBridgeApp(root)
    root.mainloop()
