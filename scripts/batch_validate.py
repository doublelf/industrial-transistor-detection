"""Batch validation over the whole MVTec AD transistor TEST set.

Runs zero-shot WinCLIP on every image, writes results/batch_results.csv,
and computes image-level AUROC against the official ground truth
(image label = directory name: 'good' -> OK, any other folder -> defect).

AUROC is computed from RAW scores, independent of the OK/NG threshold,
so it is comparable to the paper's protocol.

Usage:
    python scripts/batch_validate.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from winclip_detector import (  # noqa: E402
    ZeroShotWinCLIP,
    decide,
    load_config,
    load_threshold,
    sha256_of,
)

# Paper reference (NOT this project's measurement).
PAPER_ZERO_SHOT_I_AUROC_MVTEC_AD = 0.918  # WinCLIP paper avg over 15 MVTec AD classes


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC via Mann-Whitney U, with correct tie handling."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def collect(data_root: Path) -> list[tuple[Path, str, int]]:
    items: list[tuple[Path, str, int]] = []
    for defect_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        label = 0 if defect_dir.name == "good" else 1
        for img in sorted(defect_dir.glob("*.png")):
            items.append((img, defect_dir.name, label))
    return items


def main() -> int:
    cfg = load_config()
    data_root = PROJECT_ROOT / cfg["data"]["root"]
    if not data_root.is_dir():
        print(f"[ERROR] test set not found: {data_root}")
        return 2

    threshold = load_threshold(cfg)
    items = collect(data_root)
    print(f"[INFO] images to run: {len(items)}")
    print(f"[INFO] threshold    : {threshold:.6f} (calibrated on test/good only)")

    det = ZeroShotWinCLIP(cfg)
    print(f"[INFO] model loaded in {det.load_seconds:.2f}s | prompts={det.prompts.version} | k_shot={det.model.k_shot}")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, (path, defect_type, label) in enumerate(items, 1):
        res = det.predict_path(path)
        verdict = decide(res["score"], threshold)
        rows.append(
            {
                "image": path.name,
                "defect_type": defect_type,
                "ground_truth": "defect" if label else "good",
                "label": label,
                "score": f"{res['score']:.6f}",
                "threshold": f"{threshold:.6f}",
                "decision": verdict,
                "margin": f"{res['score'] - threshold:.6f}",
                "correct": int((verdict == "NG") == bool(label)),
                "infer_ms": f"{res['infer_ms']:.1f}",
                "map_min": f"{float(res['anomaly_map'].min()):.6f}",
                "map_max": f"{float(res['anomaly_map'].max()):.6f}",
                "prompt_version": det.prompts.version,
            }
        )
        print(f"  [{i:>3}/{len(items)}] {defect_type:<13} {path.name:<10} "
              f"score={res['score']:.6f} -> {verdict}")
    elapsed = time.perf_counter() - t0

    out_csv = PROJECT_ROOT / "results" / "batch_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] wrote {out_csv}")

    scores = np.array([float(r["score"]) for r in rows])
    labels = np.array([r["label"] for r in rows])
    auc = auroc(scores, labels)

    good = scores[labels == 0]
    def_ = scores[labels == 1]

    tp = sum(1 for r in rows if r["label"] == 1 and r["decision"] == "NG")
    fn = sum(1 for r in rows if r["label"] == 1 and r["decision"] == "OK")
    fp = sum(1 for r in rows if r["label"] == 0 and r["decision"] == "NG")
    tn = sum(1 for r in rows if r["label"] == 0 and r["decision"] == "OK")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")

    summary = {
        "n_images": len(rows),
        "n_good": int((labels == 0).sum()),
        "n_defect": int((labels == 1).sum()),
        "auroc_measured": auc,
        "auroc_paper_reference": PAPER_ZERO_SHOT_I_AUROC_MVTEC_AD,
        "threshold": threshold,
        "good_score_mean": float(good.mean()),
        "good_score_std": float(good.std(ddof=1)),
        "good_score_min": float(good.min()),
        "good_score_max": float(good.max()),
        "defect_score_mean": float(def_.mean()),
        "defect_score_min": float(def_.min()),
        "defect_score_max": float(def_.max()),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall_ng": recall,
        "precision_ng": precision,
        "accuracy": (tp + tn) / len(rows),
        "avg_infer_ms": float(np.mean([float(r["infer_ms"]) for r in rows])),
        "total_seconds": elapsed,
        "prompt_version": det.prompts.version,
    }

    print()
    print("=" * 66)
    print(" BATCH VALIDATION SUMMARY (transistor test set)")
    print("=" * 66)
    print(f"  images               : {summary['n_images']} (good {summary['n_good']} / defect {summary['n_defect']})")
    print(f"  threshold            : {threshold:.6f}  (mean + k*std on test/good)")
    print(f"  good   score mean/std: {summary['good_score_mean']:.6f} / {summary['good_score_std']:.6f}")
    print(f"  defect score mean    : {summary['defect_score_mean']:.6f}")
    print(f"  TP/FN/FP/TN          : {tp}/{fn}/{fp}/{tn}")
    print(f"  recall (defect caught): {recall:.4f}")
    print(f"  precision (NG right)  : {precision:.4f}")
    print(f"  accuracy              : {summary['accuracy']:.4f}")
    print(f"  avg infer (CPU)      : {summary['avg_infer_ms']:.0f} ms")
    print()
    print(f"  IMAGE-LEVEL AUROC  measured = {auc:.4f}")
    print(f"                     paper    = {PAPER_ZERO_SHOT_I_AUROC_MVTEC_AD:.4f} "
          f"(WinCLIP paper, avg over 15 MVTec AD classes, e31 backbone)")
    print("  -> These are DIFFERENT numbers from DIFFERENT sources; see run_report.md.")
    print("=" * 66)

    # Compact machine-readable summary
    import json

    summ_path = PROJECT_ROOT / "results" / "batch_summary.json"
    payload = dict(summary)
    payload["weights_sha256"] = sha256_of(det.weights_path)
    payload["threshold_rule"] = cfg["threshold"]["rule"]
    payload["note_auroc"] = (
        "measured = this run on transistor only; paper = WinCLIP paper average over 15 classes. "
        "Not directly comparable."
    )
    summ_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] summary -> {summ_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
