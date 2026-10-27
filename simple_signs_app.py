"""
SignBridge Simple Demo
Local webcam demo for a small set of hand poses using MediaPipe Hands.
This is a pose demo, not a validated Egyptian Sign Language translator.
"""
import tkinter as tk
from tkinter import messagebox
from collections import deque, Counter
import threading
import cv2
import mediapipe as mp
from PIL import Image, ImageTk

BG = "#0b1020"
PANEL = "#151d31"
TEXT = "#f4f7ff"
MUTED = "#a8b3cc"
ACCENT = "#5b8cff"
GREEN = "#56d6a2"
AMBER = "#f2c66d"

def fingers_up(lm):
    # Finger tip and PIP comparisons are robust enough for a basic demo when
    # the palm faces the camera. Thumb uses x-direction and can vary by hand.
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y
    # Compare thumb tip to IP joint; handedness-specific direction is handled
    # by MediaPipe handedness when available.
    return index, middle, ring, pinky

def classify_pose(hand, handedness_label="Right"):
    lm = hand.landmark
    idx, mid, ring, pinky = fingers_up(lm)
    thumb_up = lm[4].y < lm[3].y and lm[4].y < lm[2].y
    folded_count = sum(not x for x in (idx, mid, ring, pinky))

    # 👍 A clear raised thumb while the four fingers are folded.
    if thumb_up and folded_count >= 3:
        return ("👍  تمام / ممتاز", 0.82)
    # ✋ Open palm is presented as a greeting gesture for the demo. This is
    # not a claim that one static pose equals the full Arabic sign phrase.
    if idx and mid and ring and pinky:
        return ("✋  أهلاً / تحية", 0.78)
    if idx and mid and not ring and not pinky:
        return ("✌️  سلام / إشارة النصر", 0.72)
    if not idx and not mid and not ring and not pinky and not thumb_up:
        return ("✊  قبضة اليد", 0.70)
    if idx and not mid and not ring and not pinky:
        return ("☝️  السبابة", 0.68)
    return ("غير معروف — جرّب وضع اليد بوضوح", 0.0)

class SignBridgeSimple:
    def __init__(self, root):
        self.root = root
        root.title("SignBridge | عرض مبسّط")
        root.geometry("1100x760")
        root.minsize(850, 620)
        root.configure(bg=BG)
        self.cap = None
        self.running = False
        self.photo = None
        self.history = deque(maxlen=7)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            model_complexity=0, min_detection_confidence=0.55,
            min_tracking_confidence=0.55)
        self.draw = mp.solutions.drawing_utils
        self.build_ui()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def build_ui(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=22, pady=16)
        tk.Label(header, text="SignBridge", font=("Segoe UI", 27, "bold"),
                 bg=BG, fg=TEXT).pack(anchor="w")
        tk.Label(header, text="عرض تجريبي للتعرّف على أوضاع اليد — 5 حركات بسيطة",
                 font=("Segoe UI", 12), bg=BG, fg=MUTED).pack(anchor="w", pady=(4, 0))
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=22, pady=(0, 16))
        self.video = tk.Label(body, bg="#050811", text="اضغط «تشغيل الكاميرا»",
                              fg=MUTED, font=("Segoe UI", 16))
        self.video.pack(side="left", fill="both", expand=True, padx=(0, 14))
        panel = tk.Frame(body, bg=PANEL, width=300)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        tk.Label(panel, text="الإشارة المتوقعة", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 12)).pack(anchor="w", padx=18, pady=(22, 8))
        self.result = tk.Label(panel, text="جاهز", bg=PANEL, fg=TEXT,
                               font=("Segoe UI", 22, "bold"), wraplength=250,
                               justify="right")
        self.result.pack(fill="x", padx=18, pady=(0, 12))
        self.status = tk.Label(panel, text="الكاميرا متوقفة", bg=PANEL, fg=AMBER,
                               font=("Segoe UI", 10), wraplength=250, justify="right")
        self.status.pack(fill="x", padx=18, pady=(0, 16))
        tk.Label(panel, text="الحركات المتاحة", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=18, pady=(8, 8))
        for s in ["👍 تمام / ممتاز", "✋ أهلاً / تحية", "✌️ سلام / النصر",
                  "✊ قبضة اليد", "☝️ السبابة"]:
            tk.Label(panel, text=s, bg=PANEL, fg=MUTED, font=("Segoe UI", 12),
                     anchor="e").pack(fill="x", padx=18, pady=4)
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=22, pady=(0, 10))
        self.start_btn = tk.Button(controls, text="تشغيل الكاميرا", command=self.start,
                                   bg=ACCENT, fg="white", activebackground=ACCENT,
                                   relief="flat", font=("Segoe UI", 12, "bold"),
                                   padx=20, pady=10)
        self.start_btn.pack(side="left")
        tk.Button(controls, text="إيقاف", command=self.stop, bg=PANEL, fg=TEXT,
                  relief="flat", font=("Segoe UI", 12), padx=20, pady=10).pack(side="left", padx=8)
        tk.Label(self.root, text="تنبيه: هذا نموذج عرض تجريبي لوضعيات اليد، وليس مترجمًا معتمدًا للغة الإشارة المصرية. «سلام عليكم» عبارة حركية كاملة وقد لا يمكن استنتاجها من وضع ثابت واحد.",
                 bg=BG, fg=MUTED, wraplength=1050, justify="left",
                 font=("Segoe UI", 9)).pack(fill="x", padx=22, pady=(0, 14))

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
        if not self.cap.isOpened():
            if self.cap:
                self.cap.release()
            self.cap = None
            messagebox.showerror("SignBridge", "مش قادر أفتح الكاميرا. اقفل Zoom/Teams وجرب تاني.")
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.status.config(text="الكاميرا شغالة — قرّب يدك وخليها واضحة", fg=GREEN)
        self.update_frame()

    def update_frame(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status.config(text="تعذرت قراءة الكاميرا", fg=AMBER)
            self.root.after(100, self.update_frame)
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            self.draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            label, _ = classify_pose(hand)
            self.history.append(label)
            stable = Counter(self.history).most_common(1)[0][0]
            self.result.config(text=stable, fg=GREEN if "غير معروف" not in stable else AMBER)
        else:
            self.history.clear()
            self.result.config(text="مش شايف إيد — حطها قدام الكاميرا", fg=TEXT)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w, max_h = 740, 560
        image.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(image)
        self.video.config(image=self.photo, text="")
        self.root.after(25, self.update_frame)

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.status.config(text="الكاميرا متوقفة", fg=AMBER)
        self.video.config(image="", text="اضغط «تشغيل الكاميرا»")
        self.photo = None

    def close(self):
        self.stop()
        self.hands.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    SignBridgeSimple(root)
    root.mainloop()
