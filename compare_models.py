"""
SignBridge - RAW vs RELATIVE comparison (leave-one-camera-position-out).

For every camera position (CENTER, LEFT, RIGHT, TOP, BOTTOM) the script trains on
the other four positions and tests on the held-out one, for both feature sets:
  RAW      : MediaPipe x, y, z as they come out of the camera frame
  RELATIVE : coordinates relative to the wrist, scaled by hand size

Reported per model: accuracy, error rate, macro-F1, prediction latency (ms) and
training time. Each experiment is repeated with several random seeds.
A random 80/20 split is also reported ONLY to show why it is misleading:
consecutive frames of the same recording end up in both train and test.

Outputs are written to the results/ folder (CSV, Markdown summary, PNG charts
when matplotlib is installed).
"""
import csv
import math
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, "research_data.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
POSITIONS = ["CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM"]
SEEDS = [42, 43, 44]
LATENCY_SAMPLES = 100

RAW_COLS = [f"raw_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
REL_COLS = [f"rel_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
FEATURE_SETS = {"RAW": RAW_COLS, "RELATIVE": REL_COLS}


def load_rows():
    if not os.path.isfile(DATA_FILE):
        raise SystemExit(f"Data file not found: {DATA_FILE}")
    with open(DATA_FILE, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("research_data.csv has no samples.")

    valid = []
    for row in rows:
        if not row.get("label") or not row.get("position"):
            continue
        try:
            values = [float(row[c]) for c in RAW_COLS + REL_COLS]
            if len(values) == 126 and all(math.isfinite(v) for v in values):
                valid.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    if not valid:
        raise SystemExit("No valid rows with complete raw_ and rel_ features were found.")
    return valid


def make_model(seed):
    return RandomForestClassifier(
        n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1
    )


def latency_ms(model, X_test):
    """Average time to classify ONE sample (as the live app does), in ms."""
    model.set_params(n_jobs=1)
    n = min(LATENCY_SAMPLES, len(X_test))
    start = time.perf_counter()
    for i in range(n):
        model.predict(X_test[i:i + 1])
    return (time.perf_counter() - start) / n * 1000.0


def run_holdout(rows, position, feature_name, seed, labels):
    cols = FEATURE_SETS[feature_name]
    train = [r for r in rows if r["position"].upper() != position]
    test = [r for r in rows if r["position"].upper() == position]
    if not train or not test:
        return None
    y_train = [r["label"] for r in train]
    y_test = [r["label"] for r in test]
    if len(set(y_train)) < 2 or not set(y_test).issubset(set(y_train)):
        return None

    X_train = np.asarray([[float(r[c]) for c in cols] for r in train])
    X_test = np.asarray([[float(r[c]) for c in cols] for r in test])

    model = make_model(seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred) * 100
    return {
        "position": position,
        "features": feature_name,
        "seed": seed,
        "test_samples": len(test),
        "accuracy": acc,
        "error_rate": 100 - acc,
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0) * 100,
        "latency_ms": latency_ms(model, X_test),
        "train_seconds": train_s,
        "confusion": confusion_matrix(y_test, pred, labels=labels),
    }


def run_random_split(rows, feature_name, seed):
    cols = FEATURE_SETS[feature_name]
    X = np.asarray([[float(r[c]) for c in cols] for r in rows])
    y = [r["label"] for r in rows]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = make_model(seed)
    model.fit(X_tr, y_tr)
    return accuracy_score(y_te, model.predict(X_te)) * 100


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def save_plots(summary, confusions, labels, positions):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - charts skipped)")
        return []

    saved = []
    x = np.arange(len(positions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    raw = [summary[("RAW", p)]["accuracy"][0] for p in positions]
    rel = [summary[("RELATIVE", p)]["accuracy"][0] for p in positions]
    raw_e = [summary[("RAW", p)]["accuracy"][1] for p in positions]
    rel_e = [summary[("RELATIVE", p)]["accuracy"][1] for p in positions]
    ax.bar(x - width / 2, raw, width, yerr=raw_e, label="RAW", color="#D83B55", capsize=3)
    ax.bar(x + width / 2, rel, width, yerr=rel_e, label="RELATIVE", color="#58D6A5", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylabel("Accuracy on held-out position (%)")
    ax.set_ylim(0, 105)
    ax.set_title("RAW vs RELATIVE - leave-one-position-out")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "accuracy_by_position.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    for name in FEATURE_SETS:
        cm = confusions[name]
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix - {name}\n(all held-out positions, summed)")
        top = cm.max() if cm.max() else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > top / 2 else "black")
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, f"confusion_{name.lower()}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    return saved


def main():
    rows = load_rows()
    labels = sorted({r["label"] for r in rows})
    present = [p for p in POSITIONS if any(r["position"].upper() == p for r in rows)]
    sessions = len({r.get("session_id", "") for r in rows})
    print(f"Valid samples loaded: {len(rows)} | labels: {len(labels)} | "
          f"recording sessions: {sessions} | seeds: {SEEDS}")
    print("Evaluation: leave one camera position out (train on the other positions)\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_run = []
    confusions = {name: np.zeros((len(labels), len(labels)), dtype=int) for name in FEATURE_SETS}
    summary = {}

    for position in present:
        for name in FEATURE_SETS:
            runs = []
            for seed in SEEDS:
                result = run_holdout(rows, position, name, seed, labels)
                if result is None:
                    continue
                runs.append(result)
                per_run.append(result)
                if seed == SEEDS[0]:
                    confusions[name] += result["confusion"]
            if not runs:
                continue
            summary[(name, position)] = {
                "accuracy": mean_std([r["accuracy"] for r in runs]),
                "macro_f1": mean_std([r["macro_f1"] for r in runs]),
                "latency_ms": mean_std([r["latency_ms"] for r in runs]),
                "test_samples": runs[0]["test_samples"],
            }
        if ("RAW", position) in summary and ("RELATIVE", position) in summary:
            a = summary[("RAW", position)]["accuracy"]
            b = summary[("RELATIVE", position)]["accuracy"]
            n = summary[("RAW", position)]["test_samples"]
            print(f"Held-out {position:<7} (test n={n}) | RAW {a[0]:6.2f}% +/- {a[1]:.2f} | "
                  f"RELATIVE {b[0]:6.2f}% +/- {b[1]:.2f}")

    usable = [p for p in present if ("RAW", p) in summary and ("RELATIVE", p) in summary]
    if not usable:
        raise SystemExit("No usable position splits were found.")

    print("\n========== SUMMARY ==========")
    lines = []
    overall = {}
    for name in FEATURE_SETS:
        acc = [summary[(name, p)]["accuracy"][0] for p in usable]
        f1 = [summary[(name, p)]["macro_f1"][0] for p in usable]
        lat = [summary[(name, p)]["latency_ms"][0] for p in usable]
        overall[name] = {
            "accuracy": float(np.mean(acc)),
            "error": 100 - float(np.mean(acc)),
            "worst": float(np.min(acc)),
            "f1": float(np.mean(f1)),
            "latency": float(np.mean(lat)),
        }
        o = overall[name]
        line = (f"{name:<9} mean accuracy {o['accuracy']:6.2f}% | error {o['error']:6.2f}% | "
                f"worst position {o['worst']:6.2f}% | macro-F1 {o['f1']:6.2f}% | "
                f"latency {o['latency']:.2f} ms/sample")
        print(line)
        lines.append(line)
    diff = overall["RELATIVE"]["accuracy"] - overall["RAW"]["accuracy"]
    print(f"Difference (RELATIVE - RAW): {diff:+.2f} percentage points")

    print("\nRandom 80/20 split (NOT a valid generalisation test - shown for comparison):")
    random_scores = {}
    for name in FEATURE_SETS:
        scores = [run_random_split(rows, name, seed) for seed in SEEDS]
        random_scores[name] = float(np.mean(scores))
        print(f"  {name:<9} {random_scores[name]:6.2f}%")

    print("\nCaution: all data comes from the recording sessions in research_data.csv. "
          "Held-out POSITIONS are unseen, but SIGNERS, lighting and cameras are not. "
          "Treat the results as preliminary.")

    # ---- save outputs ----
    csv_path = os.path.join(RESULTS_DIR, "comparison_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["features", "held_out_position", "seed", "test_samples",
                         "accuracy_pct", "error_pct", "macro_f1_pct",
                         "latency_ms_per_sample", "train_seconds"])
        for r in per_run:
            writer.writerow([r["features"], r["position"], r["seed"], r["test_samples"],
                             f"{r['accuracy']:.2f}", f"{r['error_rate']:.2f}",
                             f"{r['macro_f1']:.2f}", f"{r['latency_ms']:.3f}",
                             f"{r['train_seconds']:.2f}"])

    md_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("# SignBridge - RAW vs RELATIVE results\n\n")
        file.write(f"Samples: {len(rows)} | sessions: {sessions} | seeds: {SEEDS}\n\n")
        file.write("| Held-out position | RAW accuracy | RELATIVE accuracy |\n|---|---|---|\n")
        for p in usable:
            a = summary[("RAW", p)]["accuracy"]
            b = summary[("RELATIVE", p)]["accuracy"]
            file.write(f"| {p} | {a[0]:.2f}% (+/-{a[1]:.2f}) | {b[0]:.2f}% (+/-{b[1]:.2f}) |\n")
        file.write("\n```\n" + "\n".join(lines) + "\n```\n")
        file.write(f"\nDifference (RELATIVE - RAW): {diff:+.2f} percentage points\n")
        file.write(f"\nRandom 80/20 split (leaky, for comparison only): RAW "
                   f"{random_scores['RAW']:.2f}% | RELATIVE {random_scores['RELATIVE']:.2f}%\n")
        file.write("\nLimitations: one signer, one camera/lighting setup; results are preliminary.\n")

    plots = save_plots(summary, confusions, labels, usable)
    print(f"\nSaved: {csv_path}\nSaved: {md_path}")
    for path in plots:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
