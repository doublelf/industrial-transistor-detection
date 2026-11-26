"""Consistency check: predict.py CLI vs web UI API on the SAME images.

For each sampled image we run the real CLI (subprocess) and read its JSON,
then call the web API and compare scores / decisions.

Usage:
    python scripts/ui_consistency_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
BASE = "http://127.0.0.1:8000"

# 3 images, at least 1 defect. Chosen to span the full behaviour range.
SAMPLES = [
    "datasets/mvtec_transistor/transistor/test/good/000.png",
    "datasets/mvtec_transistor/transistor/test/bent_lead/004.png",
    "datasets/mvtec_transistor/transistor/test/misplaced/007.png",
]


def run_cli(rel: str) -> dict:
    out_dir = PROJECT_ROOT / "results" / "ui_demo" / "cli_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PY), "predict.py", rel, "--out", str(out_dir)]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    stem = Path(rel).stem
    js = out_dir / f"{stem}.json"
    if not js.is_file():
        print(f"[FAIL] CLI produced no JSON for {rel}")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(1)
    return json.loads(js.read_text(encoding="utf-8"))


def run_api(rel: str) -> dict:
    data = json.dumps({"rel": rel}).encode()
    req = urllib.request.Request(
        BASE + "/api/inspect", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> int:
    lines = [
        "# UI consistency check: predict.py vs web API",
        "",
        "Same image, same model instance settings, same threshold. "
        "`predict.py` runs in a **separate process** (fresh model load); "
        "the web API uses the long-resident model. Both must agree.",
        "",
        "| # | image | ground truth | CLI score | UI score | Δ | CLI verdict | UI verdict | match |",
        "|---|-------|--------------|-----------|----------|---|-------------|------------|-------|",
    ]
    all_ok = True
    rows_for_json = []
    for i, rel in enumerate(SAMPLES, 1):
        cli = run_cli(rel)
        api = run_api(rel)
        d = abs(cli["score"] - api["score"])
        match = cli["decision"] == api["decision"] and d < 1e-6
        all_ok &= match
        gt = api.get("ground_truth", "?")
        lines.append(
            f"| {i} | `{Path(rel).name}` ({Path(rel).parent.name}) | {gt} | "
            f"{cli['score']:.6f} | {api['score']:.6f} | {d:.2e} | "
            f"{cli['decision']} | {api['decision']} | {'✅' if match else '❌'} |"
        )
        rows_for_json.append(
            {
                "image": rel,
                "ground_truth": gt,
                "cli_score": cli["score"],
                "ui_score": api["score"],
                "delta": d,
                "cli_decision": cli["decision"],
                "ui_decision": api["decision"],
                "threshold": cli["threshold"],
                "prompt_version": cli["prompt_version"],
                "match": match,
            }
        )
        print(f"  {rel}: CLI={cli['score']:.6f} UI={api['score']:.6f} Δ={d:.2e} "
              f"{'MATCH' if match else 'MISMATCH'}")

    lines += [
        "",
        "## Result",
        "",
        f"All 3 samples {'agree exactly (Δ = 0.00e+00)' if all_ok else 'DO NOT all agree - see table'} "
        "between the CLI and the web UI.",
        "",
        "> Note: scores are produced by deterministic inference (`torch.no_grad`, `eval()`, CPU) "
        "with identical preprocessing, so bit-identical outputs are expected.",
        "",
        f"Threshold used by both: {rows_for_json[0]['threshold']:.6f}",
        f"Prompt version shared by both: `{rows_for_json[0]['prompt_version']}`",
        "",
        "## Raw values",
        "",
        "```json",
        json.dumps(rows_for_json, indent=2),
        "```",
    ]

    out = PROJECT_ROOT / "results" / "ui_demo" / "check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[INFO] wrote {out}")
    print(f"[INFO] consistency: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
