# UI consistency check: predict.py vs web API

Same image, same model instance settings, same threshold. `predict.py` runs in a **separate process** (fresh model load); the web API uses the long-resident model. Both must agree.

| # | image | ground truth | CLI score | UI score | Δ | CLI verdict | UI verdict | match |
|---|-------|--------------|-----------|----------|---|-------------|------------|-------|
| 1 | `000.png` (good) | good | 0.419235 | 0.419235 | 0.00e+00 | OK | OK | ✅ |
| 2 | `004.png` (bent_lead) | defect | 0.503057 | 0.503057 | 0.00e+00 | NG | NG | ✅ |
| 3 | `007.png` (misplaced) | defect | 0.539175 | 0.539175 | 0.00e+00 | NG | NG | ✅ |

## Result

All 3 samples agree exactly (Δ = 0.00e+00) between the CLI and the web UI.

> Note: scores are produced by deterministic inference (`torch.no_grad`, `eval()`, CPU) with identical preprocessing, so bit-identical outputs are expected.

Threshold used by both: 0.480053
Prompt version shared by both: `v1-official-default`

## Raw values

```json
[
  {
    "image": "datasets/mvtec_transistor/transistor/test/good/000.png",
    "ground_truth": "good",
    "cli_score": 0.419235,
    "ui_score": 0.419235,
    "delta": 0.0,
    "cli_decision": "OK",
    "ui_decision": "OK",
    "threshold": 0.480053,
    "prompt_version": "v1-official-default",
    "match": true
  },
  {
    "image": "datasets/mvtec_transistor/transistor/test/bent_lead/004.png",
    "ground_truth": "defect",
    "cli_score": 0.503057,
    "ui_score": 0.503057,
    "delta": 0.0,
    "cli_decision": "NG",
    "ui_decision": "NG",
    "threshold": 0.480053,
    "prompt_version": "v1-official-default",
    "match": true
  },
  {
    "image": "datasets/mvtec_transistor/transistor/test/misplaced/007.png",
    "ground_truth": "defect",
    "cli_score": 0.539175,
    "ui_score": 0.539175,
    "delta": 0.0,
    "cli_decision": "NG",
    "ui_decision": "NG",
    "threshold": 0.480053,
    "prompt_version": "v1-official-default",
    "match": true
  }
]
```
