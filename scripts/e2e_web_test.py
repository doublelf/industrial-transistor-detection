"""End-to-end acceptance test for the web demo (>= 20 parts).

Drives the REAL HTTP API, exactly as the browser UI does, then verifies
that UI numbers and results/live_log.csv agree.

Usage:
    python scripts/e2e_web_test.py --count 24
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def get_json(path: str) -> object:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=24)
    args = ap.parse_args()

    health = get_json("/api/health")
    if not health.get("ok"):
        print(f"[FAIL] server not ready: {health}")
        return 1
    meta = get_json("/api/meta")
    manifest = get_json("/api/manifest")
    print(f"[INFO] model={meta['backbone']} thr={meta['threshold']} prompts={meta['prompt_version']}")
    print(f"[INFO] manifest={len(manifest)} images")

    post_json("/api/reset", {})
    # Clear the live log so the check compares against only this run.
    live_log = PROJECT_ROOT / "results" / "live_log.csv"
    if live_log.exists():
        live_log.unlink()

    results = []
    t0 = time.time()
    for i in range(args.count):
        item = manifest[i % len(manifest)]
        res = post_json("/api/inspect", {"rel": item["rel"]})
        results.append(res)
        print(f"  [{i + 1:>2}/{args.count}] seq={res['seq']:<3} {res['image']:<10} "
              f"{res['ground_truth']:<6} score={res['score']:.4f} -> {res['decision']:<2} "
              f"({res['infer_ms']:.0f} ms)  total={res['stats']['total']}")
    elapsed = time.time() - t0

    # ---- verify stats ----
    final = results[-1]["stats"]
    n_ok = sum(1 for r in results if r["decision"] == "OK")
    n_ng = sum(1 for r in results if r["decision"] == "NG")
    print()
    print("=" * 68)
    print(" WEB E2E ACCEPTANCE")
    print("=" * 68)
    print(f"  parts inspected      : {len(results)}  (wall {elapsed:.1f}s)")
    print(f"  UI reported total    : {final['total']}")
    print(f"  UI OK / NG           : {final['ok']} / {final['ng']}   (expected {n_ok} / {n_ng})")
    print(f"  UI NG rate           : {final['ng_rate'] * 100:.1f}%")
    print(f"  UI avg infer         : {final['avg_infer_ms']:.0f} ms")
    print(f"  history entries kept : {len(results[-1]['history'])} (limit {meta['history_limit']})")

    ok_total = final["total"] == len(results)
    ok_counts = final["ok"] == n_ok and final["ng"] == n_ng

    # ---- verify NG actually fired ----
    ng_fired = n_ng > 0
    ng_truth_ok = all(
        (r["decision"] == "NG") == (r["ground_truth"] == "defect")
        for r in results if r["ground_truth"] != "unknown"
    )

    # ---- verify live_log.csv matches ----
    log_ok = False
    log_rows = []
    if live_log.exists():
        with live_log.open("r", encoding="utf-8", newline="") as fh:
            log_rows = list(csv.DictReader(fh))
        log_ok = len(log_rows) == len(results)
        for lr, res in zip(log_rows, results, strict=False):
            if int(lr["seq"]) != res["seq"]:
                log_ok = False
            if abs(float(lr["score"]) - res["score"]) > 1e-6:
                log_ok = False
            if lr["decision"] != res["decision"]:
                log_ok = False
        logged_ng = sum(1 for r in log_rows if r["decision"] == "NG")
        logged_ok = sum(1 for r in log_rows if r["decision"] == "OK")
        print(f"  live_log.csv rows    : {len(log_rows)} (OK {logged_ok} / NG {logged_ng})")
    else:
        print("  live_log.csv rows    : MISSING")

    # ---- history limit ----
    hist_ok = len(results[-1]["history"]) <= int(meta["history_limit"])

    print()
    print(f"  stats.total matches  : {'PASS' if ok_total else 'FAIL'}")
    print(f"  OK/NG counts match   : {'PASS' if ok_counts else 'FAIL'}")
    print(f"  NG decision fired    : {'PASS' if ng_fired else 'FAIL'} ({n_ng} NGs)")
    print(f"  decisions vs truth   : {'PASS' if ng_truth_ok else 'NOTE (some mismatches, expected)'}")
    print(f"  live_log.csv matches : {'PASS' if log_ok else 'FAIL'}")
    print(f"  history <= limit     : {'PASS' if hist_ok else 'FAIL'}")
    all_core = ok_total and ok_counts and ng_fired and log_ok and hist_ok
    print(f"\n  CORE RESULT          : {'PASS' if all_core else 'FAIL'}")
    print("=" * 68)

    out = PROJECT_ROOT / "results" / "ui_demo" / "e2e_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "count": len(results),
                "wall_seconds": round(elapsed, 1),
                "final_stats": final,
                "expected_ok": n_ok,
                "expected_ng": n_ng,
                "live_log_rows": len(log_rows),
                "checks": {
                    "stats_total": ok_total,
                    "ok_ng_counts": ok_counts,
                    "ng_fired": ng_fired,
                    "live_log_matches": log_ok,
                    "history_limit": hist_ok,
                },
                "parts": [
                    {k: r[k] for k in ("seq", "image", "ground_truth", "score", "decision", "correct", "infer_ms")}
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] details -> {out}")
    return 0 if all_core else 1


if __name__ == "__main__":
    sys.exit(main())
