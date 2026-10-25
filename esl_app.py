"""
SignBridge ESL — webcam recognition using a publicly shared Egyptian Sign Language model.
Third-party model/artifacts are downloaded on first run; see THIRD_PARTY_NOTICES.md.
This is isolated from the original SIGN_1/2/3 research app.
"""
import json
import os
import queue
import threading
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model_v2.keras"
LABEL_PATH = ARTIFACTS / "label2idx.json"
TASK_PATH = ARTIFACTS / "holistic_landmarker.task"

RAW_BASE = "https://raw.githubusercontent.com/abdullahsherdy/ESL-software-ml/main/artifacts/"
DOWNLOADS = {
    MODEL_PATH: (RAW_BASE + "model_v2.keras", 100_000),
    LABEL_PATH: (RAW_BASE + "label2idx.json", 100),
    TASK_PATH: (RAW_BASE + "holistic_landmarker.task", 1_000_000),
}
FACE_IDX = [0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NEUTRAL_IDX = EMOTION_CLASSES.index("neutral")

# Arabic display aliases for labels whose transliteration is reasonably clear.
# Unmapped labels are shown exactly as stored in the source label map.
ARABIC_LABELS = {
    "3amel eh": "عامل إيه؟",
    "3ieb": "عيب",
    "3rabia": "عربية",
    "akhoia": "أخويا",
    "alam": "ألم",
    "bkrh": "بكره",
    "bsor3a": "بسرعة",
    "delwaty": "دلوقتي",
    "docktor": "دكتور",
    "efrdha": "افردها",
    "ehtrem nfsk": "احترم نفسك",
    "elnhrda": "النهارده",
    "ermy wra dhrk": "ارمي ورا ضهرك",
    "fe sark": "في سرك",
    "flos": "فلوس",
    "harmy": "حرامي",
    "helw awy": "حلو أوي",
    "kdab": "كداب",
    "khaif": "خايف",
    "khaly balk": "خلي بالك",
    "khatbty": "خطيبتي",
    "kolya": "كلية",
    "ktab": "كتاب",
    "mafish": "مفيش",
    "manzel": "منزل",
    "mashy": "ماشي",
    "meraty": "مراتي",
    "mohands": "مهندس",
    "msh 3waz": "مش عاوز",
    "oda": "أوضة",
    "sabhl elkheer": "صباح الخير",
    "sahby": "صاحبي",
    "shar3": "شارع",
    "shbak": "شباك",
    "t3ban": "تعبان",
    "tmam": "تمام",
    "whashny": "وحشتني",
    "whda whda": "واحدة واحدة",
    "yabky": "يبكي",
    "yadhk": "يضحك",
    "yallhwy": "يا لهوي",
    "yshtm": "يشتم",
}

BG = "#0B1020"
PANEL = "#151D31"
ACCENT = "#5B8CFF"
TEXT = "#F4F7FF"
MUTED = "#A8B3CC"
GREEN = "#56D6A2"
AMBER = "#F2C66D"


def download_artifacts():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for path, (url, min_size) in DOWNLOADS.items():
        if path.exists() and path.stat().st_size >= min_size:
            continue
        tmp = path.with_suffix(path.suffix + ".download")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SignBridge/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response, open(tmp, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if tmp.stat().st_size < min_size:
                raise RuntimeError(f"Downloaded file looks incomplete: {path.name}")
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def normalize_landmarks(raw):
    raw = raw.astype(np.float64)
    left = raw[0:63].reshape(21, 3).copy()
    right = raw[63:126].reshape(21, 3).copy()
    face = raw[126:156].reshape(10, 3).copy()

    if left.any():
        left -= left[0]
        scale = np.max(np.linalg.norm(left, axis=1))
        if scale > 0:
            left /= scale
    if right.any():
        right -= right[0]
        scale = np.max(np.linalg.norm(right, axis=1))
        if scale > 0:
            right /= scale
    if face.any():
        face -= face.mean(axis=0)
        scale = np.max(np.linalg.norm(face, axis=1))
        if scale > 0:
            face /= scale
    return np.concatenate([left.flatten(), right.flatten(), face.flatten()]).astype(np.float32)


def collect_features(result):
    values = []
    for attr in ("left_hand_landmarks", "right_hand_landmarks"):
        landmarks = getattr(result, attr, None)
        if landmarks:
            for point in landmarks:
                values.extend([point.x, point.y, point.z])
        else:
            values.extend([0.0] * 63)

    face = getattr(result, "face_landmarks", None)
    if face:
        for idx in FACE_IDX:
            point = face[idx]
            values.extend([point.x, point.y, point.z])
    else:
        values.extend([0.0] * 30)

    raw = np.asarray(values, dtype=np.float32)
    if raw.shape != (156,):
        raise ValueError(f"Unexpected landmark feature shape: {raw.shape}")
    return raw


def has_hand(raw):
    return bool(np.any(raw[:126] != 0))


def draw_skeleton(frame, points):
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
    ]
    for hand in points:
        if not hand:
            continue
        xy = [(int(x*w), int(y*h)) for x,y in hand]
        for a,b in connections:
            cv2.line(frame, xy[a], xy[b], (70, 210, 150), 2, cv2.LINE_AA)
        for x,y in xy:
            cv2.circle(frame, (x,y), 3, (255,255,255), -1, cv2.LINE_AA)


class SignBridgeESLApp:
    def __init__(self, root):
        self.root = root
        root.title("SignBridge | Egyptian Sign Language")
        root.geometry("1080x720")
        root.minsize(850, 600)
        root.configure(bg=BG)

        self.stop_event = threading.Event()
        self.worker = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.state_text = "اضغط «تشغيل النظام» لتحميل النموذج وفتح الكاميرا"
        self.prediction = "—"
        self.label_source = "—"
        self.confidence = 0.0
        self.running = False
        self.photo = None

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=22, pady=(18, 8))
        tk.Label(header, text="SignBridge", font=("Segoe UI", 25, "bold"),
                 bg=BG, fg=TEXT).pack(side="left")
        tk.Label(header, text="EGYPTIAN SIGN LANGUAGE • 56 LABELS IN MODEL",
                 font=("Segoe UI", 9, "bold"), bg=BG, fg=GREEN).pack(side="right", pady=12)

        tk.Label(root, text="التعرّف على إشارات من نموذج لغة الإشارة المصرية",
                 font=("Segoe UI", 15, "bold"), bg=BG, fg=TEXT).pack(anchor="w", padx=24, pady=(0, 12))

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=22, pady=(0, 14))
        video_panel = tk.Frame(body, bg=PANEL, padx=12, pady=12)
        video_panel.pack(side="left", fill="both", expand=True)
        tk.Label(video_panel, text="LIVE CAMERA", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", pady=(0, 8))
        self.video_label = tk.Label(video_panel, text="الكاميرا لم تبدأ بعد",
                                    font=("Segoe UI", 14), bg="#060914", fg=MUTED)
        self.video_label.pack(fill="both", expand=True)

        side = tk.Frame(body, bg=PANEL, width=285, padx=18, pady=18)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)
        tk.Label(side, text="النتيجة الحالية", font=("Segoe UI", 13, "bold"),
                 bg=PANEL, fg=TEXT).pack(anchor="w")
        self.pred_label = tk.Label(side, text="—", font=("Segoe UI", 22, "bold"),
                                   bg=PANEL, fg=GREEN, wraplength=240, justify="left")
        self.pred_label.pack(anchor="w", pady=(14, 4))
        self.source_label = tk.Label(side, text="Model label: —", font=("Segoe UI", 9),
                                     bg=PANEL, fg=MUTED, wraplength=240, justify="left")
        self.source_label.pack(anchor="w", pady=(0, 10))
        self.conf_label = tk.Label(side, text="ثقة النموذج: —", font=("Segoe UI", 10),
                                   bg=PANEL, fg=MUTED)
        self.conf_label.pack(anchor="w", pady=(0, 14))
        self.status_label = tk.Label(side, text=self.state_text, font=("Segoe UI", 10),
                                     bg=PANEL, fg=AMBER, wraplength=240, justify="left")
        self.status_label.pack(anchor="w", pady=(0, 18))
        self.start_button = tk.Button(side, text="▶  تشغيل النظام", command=self.start,
                                      bg=ACCENT, fg="white", activebackground=ACCENT,
                                      activeforeground="white", relief="flat",
                                      font=("Segoe UI", 10, "bold"), pady=10, cursor="hand2")
        self.start_button.pack(fill="x", pady=4)
        tk.Button(side, text="■  إيقاف النظام", command=self.stop,
                  bg="#30394C", fg="white", activebackground="#30394C",
                  activeforeground="white", relief="flat", font=("Segoe UI", 10, "bold"),
                  pady=10, cursor="hand2").pack(fill="x", pady=4)
        tk.Label(side,
                 text="تنبيه: النموذج بحثي ويتعرف على مفردات معزولة فقط. النتيجة توقع وليست ترجمة مضمونة.",
                 font=("Segoe UI", 9), bg=PANEL, fg=AMBER, wraplength=240,
                 justify="left").pack(anchor="w", pady=(18, 0))

        tk.Label(root, text="Model source: ESL-software-ml • attribution and limitations in THIRD_PARTY_NOTICES.md",
                 font=("Segoe UI", 8), bg=BG, fg=MUTED).pack(pady=(0, 10))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(30, self.refresh_ui)

    def update_state(self, text):
        with self.lock:
            self.state_text = text

    def start(self):
        if self.running or (self.worker and self.worker.is_alive()):
            return
        self.stop_event.clear()
        self.running = True
        self.start_button.config(state="disabled")
        self.update_state("جارٍ تنزيل ملفات النموذج المطلوبة (أول مرة فقط)...")
        self.worker = threading.Thread(target=self.worker_main, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.update_state("جارٍ إيقاف الكاميرا...")
        self.start_button.config(state="normal")

    def worker_main(self):
        cap = None
        try:
            self.update_state("جارٍ تجهيز النموذج وملفاته...")
            download_artifacts()
            from src.inference import load_model, predict_frame, get_last_mp_results

            # Use the upstream project's official preprocessing/inference path.
            # This avoids subtle feature-order and normalization mismatches.
            load_model(str(MODEL_PATH), str(LABEL_PATH))

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError("تعذر فتح الكاميرا. اقفل أي برنامج تاني بيستخدمها.")

            with open(LABEL_PATH, "r", encoding="utf-8") as f:
                label2idx = json.load(f)
            self.update_state(f"النموذج جاهز • {len(label2idx)} فئة • الكاميرا تعمل")

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.03)
                    continue
                frame = cv2.flip(frame, 1)
                # predict_frame performs the same resize, landmark extraction,
                # normalization and feature assembly used by the upstream model.
                label, confidence, emotion = predict_frame(
                    frame, cached_emotion="neutral", raw=True
                )

                small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                results = get_last_mp_results()
                latest_points = [None, None]
                if results is not None:
                    for idx, attr in enumerate(("left_hand_landmarks", "right_hand_landmarks")):
                        hand = getattr(results, attr, None)
                        latest_points[idx] = [(p.x, p.y) for p in hand] if hand else None
                draw_skeleton(small, latest_points)

                if label == "__no_hands__":
                    shown = "لم يتم رصد يد"
                    label_for_ui = "لم يتم رصد يد"
                    conf_for_ui = 0.0
                else:
                    conf_for_ui = float(confidence)
                    # A conservative display gate hides low-score guesses.
                    # This is not calibrated accuracy and cannot catch every confident error.
                    if conf_for_ui < 0.70:
                        shown = "إشارة غير مؤكدة"
                        label_for_ui = shown
                    else:
                        shown = ARABIC_LABELS.get(label, label)
                        label_for_ui = shown
                    cv2.rectangle(small, (0, 0), (320, 58), (11, 16, 32), -1)
                    cv2.putText(small, str(label)[:25], (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                                0.48, (86, 214, 162), 2, cv2.LINE_AA)
                    cv2.putText(small, f"{conf_for_ui * 100:.0f}% model score", (8, 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 245, 255), 1, cv2.LINE_AA)

                display = cv2.resize(small, (640, 480), interpolation=cv2.INTER_LINEAR)
                with self.lock:
                    self.latest_frame = display
                    self.prediction = label_for_ui
                    self.label_source = label
                    self.confidence = conf_for_ui
        except Exception as exc:
            self.update_state(f"خطأ: {exc}")
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                self.state_text = ("النظام متوقف" if self.stop_event.is_set()
                                   else "توقف التشغيل بسبب خطأ أو انتهاء المعالجة.")
                self.running = False

    def refresh_ui(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            state = self.state_text
            prediction = self.prediction
            source_label = self.label_source
            confidence = self.confidence
            running = self.running
        self.start_button.config(state="disabled" if running else "normal")
        self.status_label.config(text=state)
        self.pred_label.config(text=prediction)
        self.source_label.config(text=f"Model label: {source_label}")
        self.conf_label.config(text=f"درجة ثقة النموذج: {confidence * 100:.1f}%" if confidence else "درجة ثقة النموذج: —")
        if frame is not None:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=image)
            self.video_label.config(image=self.photo, text="")
        self.root.after(30, self.refresh_ui)

    def close(self):
        self.stop_event.set()
        self.root.after(150, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    SignBridgeESLApp(root)
    root.mainloop()
