r"""
SignBridge pre-flight check - run this on the presentation laptop BEFORE the demo.

    .venv\Scripts\python.exe preflight.py

Checks Python/libraries, model files, training data and the camera, and prints
a clear OK / FAIL line for each. Exit code 0 means ready.
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
failures = []


def report(ok, text, hint=""):
    print(("[ OK ] " if ok else "[FAIL] ") + text)
    if not ok:
        failures.append(text)
        if hint:
            print("       -> " + hint)


print("SignBridge pre-flight check\n")
report(sys.version_info[:2] == (3, 11), f"Python {sys.version.split()[0]} (3.11 recommended)",
       "Run setup.bat to create the .venv with Python 3.11")

for module in ["cv2", "mediapipe", "numpy", "sklearn", "joblib", "PIL", "tkinter", "pyttsx3"]:
    try:
        importlib.import_module(module)
        report(True, f"library {module}")
    except Exception as error:  # noqa: BLE001
        optional = module == "pyttsx3"
        report(optional, f"library {module}" + (" (optional - voice disabled)" if optional else ""),
               f"Missing/broken: {error}. Run setup.bat again.")

try:
    import joblib
    import numpy as np
    model = joblib.load(os.path.join(ROOT, "model_relative.pkl"))
    ok = getattr(model, "n_features_in_", None) == 63
    report(ok, f"model_relative.pkl loads, classes: {[str(c) for c in model.classes_]}",
           "Expected a model with 63 relative features")
    if ok:
        sample = np.zeros((1, 63), dtype=np.float32)
        model.predict(sample)
        report(True, "model produces a prediction")
except Exception as error:  # noqa: BLE001
    report(False, "model_relative.pkl", str(error))

data_file = os.path.join(ROOT, "research_data.csv")
report(os.path.isfile(data_file), "research_data.csv present", "Needed only for retraining/comparison")

try:
    import cv2
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    opened = None
    for index in (0, 1):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                opened = index
                break
        else:
            cap.release()
    report(opened is not None, f"camera works (index {opened})" if opened is not None else "camera",
           "Close Zoom/Teams/browser tabs using the camera and allow camera access in Windows Settings")
except Exception as error:  # noqa: BLE001
    report(False, "camera", str(error))

print()
if failures:
    print(f"{len(failures)} problem(s) found - fix them before the demo.")
    sys.exit(1)
print("All checks passed. Ready for the demo.")
