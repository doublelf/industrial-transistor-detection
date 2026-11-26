"""Derive the OK/NG threshold from NORMAL test images only.

Rule (declared in advance, never tuned on defects):
    threshold = mean(scores on test/good) + k * std(scores on test/good)

The script only ever reads ``test/good``. Defect subfolders are not opened.

Usage:
    python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from winclip_detector import ZeroShotWinCLIP, load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    data_root = PROJECT_ROOT / cfg["data"]["root"]
    good_dir = data_root / cfg["data"]["good_dir"]
    if not good_dir.is_dir():
        print(f"[ERROR] good-image directory not found: {good_dir}")
        return 2

    good_images = sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg"))
    if not good_images:
        print(f"[ERROR] no images in {good_dir}")
        return 2

    print(f"[INFO] calibrating on {len(good_images)} NORMAL images from {good_dir}")
    print("[INFO] defect directories are deliberately NOT read by this script")

    det = ZeroShotWinCLIP(cfg)
    print(f"[INFO] model loaded in {det.load_seconds:.2f}s | prompts={det.prompts.version}")

    scores: list[float] = []
    timings: list[float] = []
    t_all = time.perf_counter()
    for i, path in enumerate(good_images, 1):
        res = det.predict_path(path)
        scores.append(res["score"])
        timings.append(res["infer_ms"])
        print(f"  [{i:>3}/{len(good_images)}] {path.name:<24} score={res['score']:.6f} ({res['infer_ms']:.0f} ms)")

    arr = np.asarray(scores, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    k = float(cfg["threshold"]["k"])
    thr = mean + k * std

    print()
    print("=" * 62)
    print("NORMAL-ONLY score distribution (test/good)")
    print("=" * 62)
    print(f"  n            = {arr.size}")
    print(f"  mean         = {mean:.6f}")
    print(f"  std (ddof=1) = {std:.6f}")
    print(f"  min / max    = {arr.min():.6f} / {arr.max():.6f}")
    print(f"  p50 / p95    = {np.percentile(arr, 50):.6f} / {np.percentile(arr, 95):.6f}")
    print(f"  rule         = mean + {k} * std")
    print(f"  THRESHOLD    = {thr:.6f}")
    print(f"  good images above threshold (false NG by construction): {int((arr >= thr).sum())}")
    print(f"  total inference time: {time.perf_counter() - t_all:.1f}s "
          f"(avg {np.mean(timings):.0f} ms/image)")
    print("=" * 62)

    # Write back into config.yaml (text patch, keeps comments/formatting).
    cfg_path = Path(cfg["_config_path"])
    text = cfg_path.read_text(encoding="utf-8")
    stamp = datetime.now().isoformat(timespec="seconds")
    replacements = {
        "  value: ": None if False else f"{thr:.6f}",
        "  n_good: ": str(arr.size),
        "  mean: ": f"{mean:.6f}",
        "  std: ": f"{std:.6f}",
        "  calibrated_at: ": f'"{stamp}"',
    }
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        for key, val in replacements.items():
            if line.startswith(key):
                lines[idx] = f"{key}{val}"
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] threshold written to {cfg_path}")

    # Keep a machine-readable copy for the report.
    out = PROJECT_ROOT / "results" / "threshold_calibration.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                f"prompt_version={det.prompts.version}",
                f"class_name={det.prompts.class_name}",
                f"n_good={arr.size}",
                f"mean={mean:.6f}",
                f"std_ddof1={std:.6f}",
                f"k={k}",
                f"threshold={thr:.6f}",
                f"good_min={arr.min():.6f}",
                f"good_max={arr.max():.6f}",
                f"good_false_ng={int((arr >= thr).sum())}",
                f"source={good_dir}",
                f"calibrated_at={stamp}",
                "",
                "per_image:",
                *[f"{p.name},{s:.6f}" for p, s in zip(good_images, scores, strict=True)],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] details saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
