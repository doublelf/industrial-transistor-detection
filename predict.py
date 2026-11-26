"""Single-image zero-shot anomaly inspection (WinCLIP, k_shot=0).

Outputs, for ONE image:
    * anomaly score (image level)
    * OK / NG decision against the threshold calibrated on test/good
    * heatmap overlay PNG in outputs/
    * a same-named JSON with all numbers needed to reproduce the decision

No training, no fine-tuning, no memory bank, no few-shot.

Usage:
    python predict.py <image_path> [--out outputs] [--no-heatmap]
    python predict.py datasets/mvtec_transistor/transistor/test/bent_lead/000.png
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from winclip_detector import (  # noqa: E402
    MissingWeightsError,
    ZeroShotWinCLIP,
    decide,
    load_config,
    load_threshold,
    overlay_heatmap,
    sha256_of,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Zero-shot WinCLIP single-image inspection")
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--config", default=None, help="config yaml (default: ./config.yaml)")
    ap.add_argument("--out", default=None, help="output directory (default: outputs/)")
    ap.add_argument("--no-heatmap", action="store_true", help="skip heatmap rendering")
    args = ap.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[ERROR] image not found: {image_path}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / cfg["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        threshold = load_threshold(cfg)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 3

    try:
        detector = ZeroShotWinCLIP(cfg)
    except MissingWeightsError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 4

    print(f"[INFO] backbone      : {cfg['model']['backbone']}")
    print(f"[INFO] weights sha256: {sha256_of(detector.weights_path)[:16]}...")
    print(f"[INFO] prompts       : {detector.prompts.version} "
          f"(normal={detector.prompts.n_normal}, anomalous={detector.prompts.n_anomalous})")
    print(f"[INFO] k_shot        : {detector.model.k_shot} (zero-shot)")
    print(f"[INFO] device        : {detector.device}")

    result = detector.predict_path(image_path)
    score = result["score"]
    verdict = decide(score, threshold)

    heatmap_path = None
    if not args.no_heatmap:
        heatmap_path = out_dir / f"{image_path.stem}_heatmap.png"
        overlay_heatmap(
            image_path,
            result["anomaly_map"],
            heatmap_path,
            score=score,
            threshold=threshold,
            decision=verdict,
        )
        print(f"[INFO] heatmap       : {heatmap_path}")

    metadata = detector.metadata()
    payload = {
        "image": str(image_path),
        "image_size": result["image_size"],
        "score": round(score, 6),
        "threshold": round(threshold, 6),
        "margin": round(score - threshold, 6),
        "decision": verdict,
        "decision_rule": "NG if score >= threshold",
        "threshold_rule": cfg["threshold"]["rule"],
        "threshold_k": cfg["threshold"]["k"],
        "threshold_source": cfg["threshold"]["source_split"],
        "infer_ms": result["infer_ms"],
        "total_ms": result["total_ms"],
        "anomaly_map_shape": list(result["anomaly_map"].shape),
        "anomaly_map_min": round(float(result["anomaly_map"].min()), 6),
        "anomaly_map_max": round(float(result["anomaly_map"].max()), 6),
        "heatmap_path": str(heatmap_path) if heatmap_path else None,
        "prompt_version": detector.prompts.version,
        "class_name": detector.prompts.class_name,
        "n_normal_prompts": detector.prompts.n_normal,
        "n_anomalous_prompts": detector.prompts.n_anomalous,
        "model": {
            "backbone": metadata["backbone"],
            "temperature": metadata["temperature"],
            "scales": metadata["scales"],
            "input_size": metadata["input_size"],
            "k_shot": metadata["k_shot"],
            "weights_sha256": metadata["weights_sha256"],
            "weights_file": Path(metadata["weights_path"]).name,
            "device": metadata["device"],
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "caveat": "Zero-shot score on MVTec AD test imagery; not a production defect rate.",
    }

    json_path = out_dir / f"{image_path.stem}.json"
    write_json(payload, json_path)

    print()
    print("=" * 56)
    print(f"  image      : {image_path.name}")
    print(f"  score      : {score:.6f}")
    print(f"  threshold  : {threshold:.6f}  (mean + {cfg['threshold']['k']}*std on test/good)")
    print(f"  decision   : {verdict}")
    print(f"  infer time : {result['infer_ms']:.0f} ms")
    print(f"  json       : {json_path}")
    print("=" * 56)
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
