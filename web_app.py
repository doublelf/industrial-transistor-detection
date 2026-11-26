"""FastAPI backend for the simulated production-line anomaly inspection demo.

Model is loaded ONCE at startup and kept in memory; every request reuses it.

Endpoints
    GET  /                     -> single-page UI (native HTML, no build step)
    GET  /api/meta             -> model/threshold/prompt metadata
    GET  /api/manifest         -> demo image list (mixed good + defect)
    POST /api/inspect          -> inspect ONE image by path (path must be in demo set)
    POST /api/inspect/upload   -> inspect an uploaded image (camera mode)
    GET  /batch/<name>         -> serve original / heatmap PNGs
    GET  /api/log              -> recent entries from results/live_log.csv
    POST /api/reset            -> clear session statistics
    GET  /api/health           -> liveness

IMPORTANT: this is a SIMULATED line over MVTec AD TEST imagery. Scores and NG
rates here are NOT production-representative.
"""

from __future__ import annotations

import csv
import io
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from winclip_detector import (  # noqa: E402
    MissingWeightsError,
    ZeroShotWinCLIP,
    decide,
    load_config,
    render_overlay as overlay_heatmap_bare,
)

cfg = load_config()
WEB_DIR = PROJECT_ROOT / "web"
BATCH_DIR = PROJECT_ROOT / "outputs" / "ui_batch"
LIVE_LOG = PROJECT_ROOT / cfg["paths"]["live_log"]
DATA_ROOT = PROJECT_ROOT / cfg["data"]["root"]

app = FastAPI(title="Zero-shot WinCLIP line QC demo", version="1.0.0")

_state: dict[str, Any] = {
    "detector": None,
    "model_meta": {},
    "threshold": None,
    "manifest": [],
    "session": {"total": 0, "ok": 0, "ng": 0, "infer_ms_sum": 0.0},
    "history": [],
    "lock": threading.Lock(),
    "log_written_header": False,
    "started_at": None,
}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def _build_manifest() -> list[dict[str, Any]]:
    """Mixed good/defect test images, interleaved for a realistic line feed."""
    items: list[dict[str, Any]] = []
    if not DATA_ROOT.is_dir():
        return items
    for d in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        label = "good" if d.name == "good" else "defect"
        for img in sorted(d.glob("*.png")):
            items.append(
                {
                    "rel": str(img.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "name": img.name,
                    "defect_type": d.name,
                    "ground_truth": label,
                }
            )
    return items


def _write_log_header() -> None:
    LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LIVE_LOG.exists() or LIVE_LOG.stat().st_size == 0:
        with LIVE_LOG.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                [
                    "seq", "timestamp", "image", "defect_type", "ground_truth",
                    "score", "threshold", "decision", "correct_if_truth_known",
                    "infer_ms", "prompt_version",
                ]
            )


def _append_log(row: dict[str, Any]) -> None:
    _write_log_header()
    with LIVE_LOG.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            [
                row["seq"], row["timestamp"], row["image"], row["defect_type"],
                row["ground_truth"], row["score"], row["threshold"], row["decision"],
                row["correct"], row["infer_ms"], row["prompt_version"],
            ]
        )


@app.on_event("startup")
def _startup() -> None:
    try:
        det = ZeroShotWinCLIP(cfg)
    except MissingWeightsError as exc:
        # Fail loudly but keep the process alive so the UI can show the reason.
        _state["model_error"] = str(exc)
        print(f"[FATAL] {exc}", file=sys.stderr)
        return
    thr = cfg["threshold"]["value"]
    if thr is None:
        _state["model_error"] = "Threshold not calibrated. Run scripts/calibrate_threshold.py first."
        return
    _state["detector"] = det
    _state["threshold"] = float(thr)
    _state["model_meta"] = det.metadata()
    _state["manifest"] = _build_manifest()
    _state["started_at"] = datetime.now().isoformat(timespec="seconds")
    _write_log_header()
    print(f"[INFO] model ready in {det.load_seconds:.1f}s | thr={thr:.6f} "
          f"| demo images={len(_state['manifest'])} | prompts={det.prompts.version}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = WEB_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(500, "web/index.html missing")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": _state["detector"] is not None, "error": _state.get("model_error")}


@app.get("/api/meta")
def meta() -> JSONResponse:
    if _state["detector"] is None:
        return JSONResponse({"error": _state.get("model_error", "model not ready")}, status_code=503)
    m = dict(_state["model_meta"])
    m["threshold"] = _state["threshold"]
    m["threshold_rule"] = cfg["threshold"]["rule"]
    m["threshold_k"] = cfg["threshold"]["k"]
    m["threshold_source"] = cfg["threshold"]["source_split"]
    m["n_demo_images"] = len(_state["manifest"])
    m["cycle_seconds_default"] = cfg["server"]["cycle_seconds"]
    m["history_limit"] = cfg["server"]["history_limit"]
    m["device"] = m.get("device", "cpu")
    m["disclaimer"] = "SIMULATION: images are from the MVTec AD test set. Zero-shot scores and NG rate do NOT represent real production performance."
    return JSONResponse(m)


@app.get("/api/manifest")
def manifest() -> JSONResponse:
    return JSONResponse(_state["manifest"])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _infer(image_path: Path, display_name: str, defect_type: str, ground_truth: str) -> dict[str, Any]:
    det: ZeroShotWinCLIP = _state["detector"]
    threshold: float = _state["threshold"]

    res = det.predict_path(image_path)
    score = float(res["score"])
    verdict = decide(score, threshold)

    with _state["lock"]:
        sess = _state["session"]
        sess["total"] += 1
        sess["ok" if verdict == "OK" else "ng"] += 1
        sess["infer_ms_sum"] += float(res["infer_ms"])
        seq = sess["total"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    stem = f"{seq:05d}_{ts}_{image_path.stem}"
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    original_rel = f"ui_batch/{stem}__original.png"
    heatmap_rel = f"ui_batch/{stem}__heatmap.png"

    with Image.open(image_path) as im:
        im.convert("RGB").save(PROJECT_ROOT / "outputs" / original_rel)
    # The UI displays a BARE overlay (same aspect as the photo, no axes or
    # colorbar) so it fills its tile edge to edge. Score/threshold/verdict
    # are rendered by the HTML around it, not baked into the image.
    overlay_heatmap_bare(
        image_path,
        res["anomaly_map"],
        PROJECT_ROOT / "outputs" / heatmap_rel,
        threshold=threshold,
    )

    truth_known = ground_truth in ("good", "defect")
    correct = None
    if truth_known:
        correct = int((verdict == "NG") == (ground_truth == "defect"))

    row = {
        "seq": seq,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "image": display_name,
        "defect_type": defect_type,
        "ground_truth": ground_truth,
        "score": f"{score:.6f}",
        "threshold": f"{threshold:.6f}",
        "decision": verdict,
        "correct": "" if correct is None else correct,
        "infer_ms": f"{res['infer_ms']:.1f}",
        "prompt_version": det.prompts.version,
    }
    _append_log(row)

    entry = {
        "seq": seq,
        "image": display_name,
        "defect_type": defect_type,
        "ground_truth": ground_truth,
        "score": round(score, 6),
        "threshold": round(threshold, 6),
        "margin": round(score - threshold, 6),
        "decision": verdict,
        "correct": correct,
        "infer_ms": round(float(res["infer_ms"]), 1),
        "original_url": f"/batch/{Path(original_rel).name}",
        "heatmap_url": f"/batch/{Path(heatmap_rel).name}",
        "timestamp": row["timestamp"],
    }
    with _state["lock"]:
        hist = _state["history"]
        hist.append(entry)
        limit = int(cfg["server"]["history_limit"])
        while len(hist) > limit:
            hist.pop(0)

    sess = _state["session"]
    return {
        **entry,
        "stats": {
            "total": sess["total"],
            "ok": sess["ok"],
            "ng": sess["ng"],
            "ng_rate": (sess["ng"] / sess["total"]) if sess["total"] else 0.0,
            "avg_infer_ms": (sess["infer_ms_sum"] / sess["total"]) if sess["total"] else 0.0,
        },
        "history": list(hist),
    }


def _require_model() -> None:
    if _state["detector"] is None:
        raise HTTPException(503, _state.get("model_error", "model not ready"))


@app.post("/api/inspect")
async def inspect(payload: dict[str, Any]) -> JSONResponse:
    _require_model()
    rel = payload.get("rel")
    if not rel:
        raise HTTPException(400, "missing 'rel'")
    entry = next((m for m in _state["manifest"] if m["rel"] == rel), None)
    if entry is None:
        raise HTTPException(400, "image not part of the demo manifest")
    path = PROJECT_ROOT / entry["rel"]
    if not path.is_file():
        raise HTTPException(404, f"image missing on disk: {entry['rel']}")
    out = _infer(path, entry["name"], entry["defect_type"], entry["ground_truth"])
    return JSONResponse(out)


@app.post("/api/inspect/upload")
async def inspect_upload(file: UploadFile) -> JSONResponse:
    """Camera-mode capture. Truth is unknown, so no correctness is scored."""
    _require_model()
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"cannot decode image: {exc}") from exc

    tmp_dir = PROJECT_ROOT / "outputs" / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"cam_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"
    img.save(tmp)

    out = _infer(tmp, tmp.name, "camera_capture", "unknown")
    return JSONResponse(out)


@app.get("/batch/{filename}")
def batch_file(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")
    path = BATCH_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/log")
def read_log(limit: int = 50) -> JSONResponse:
    if not LIVE_LOG.is_file():
        return JSONResponse([])
    with LIVE_LOG.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return JSONResponse(rows[-max(1, min(limit, 500)):])


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    with _state["lock"]:
        _state["session"] = {"total": 0, "ok": 0, "ng": 0, "infer_ms_sum": 0.0}
        _state["history"] = []
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=cfg["server"]["host"],
        port=int(cfg["server"]["port"]),
        log_level="info",
    )


# Keep random import used for potential shuffling in future; explicit no-op.
_ = random
