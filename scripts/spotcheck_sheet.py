"""Build a spot-check sheet for 6 sampled images (visual inspection aid).

Writes results/ui_demo/spotcheck_<i>.png (input + heatmap side by side) and
prints a table to paste into run_report.md.

Usage:
    python scripts/spotcheck_sheet.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from winclip_detector import ZeroShotWinCLIP, load_config, overlay_heatmap  # noqa: E402

# Deliberately mixes: caught defect, missed defect, false-NG good, easy good,
# and covers all 4 defect types where possible.
SAMPLES = [
    ("good", "000.png"),           # true good, lowest score  -> expect OK
    ("damaged_case", "001.png"),   # caught defect            -> expect NG
    ("misplaced", "007.png"),      # caught defect            -> expect NG
    ("cut_lead", "005.png"),       # MISSED defect (hardest)  -> expect OK (wrong)
    ("bent_lead", "002.png"),      # MISSED defect            -> expect OK (wrong)
    ("good", "052.png"),           # false NG (good but high) -> expect NG (wrong)
]


def main() -> int:
    cfg = load_config()
    threshold = float(cfg["threshold"]["value"])
    data_root = PROJECT_ROOT / cfg["data"]["root"]
    out_dir = PROJECT_ROOT / "results" / "ui_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {
        (r["defect_type"], r["image"]): r
        for r in csv.DictReader((PROJECT_ROOT / "results" / "batch_results.csv").open(encoding="utf-8"))
    }

    det = ZeroShotWinCLIP(cfg)
    print(f"{'#':<3}{'folder':<14}{'image':<10}{'truth':<8}{'score':<10}{'thr':<10}{'decision':<10}{'correct'}")
    print("-" * 72)
    for i, (folder, name) in enumerate(SAMPLES, 1):
        path = data_root / folder / name
        res = det.predict_path(path)
        row = rows[(folder, name)]
        assert abs(float(row["score"]) - res["score"]) < 1e-6, "batch CSV disagrees with fresh inference"
        out = out_dir / f"spotcheck_{i}.png"
        overlay_heatmap(
            path, res["anomaly_map"], out,
            score=res["score"], threshold=threshold, decision=row["decision"],
        )
        print(f"{i:<3}{folder:<14}{name:<10}{row['ground_truth']:<8}"
              f"{res['score']:<10.4f}{threshold:<10.4f}{row['decision']:<10}{row['correct']}  -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
