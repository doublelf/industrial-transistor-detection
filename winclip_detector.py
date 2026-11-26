"""Zero-shot WinCLIP inference core (project scope: k_shot = 0 only).

This module wraps anomalib's official ``WinClipModel`` without modifying the
installed package on disk. The single monkeypatch redirects the pretrained
weight tag to a LOCAL file path, which makes ``open_clip.create_model`` take
the ``os.path.exists`` branch instead of any download path:

    open_clip/factory.py:281-290
        pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        if pretrained_cfg:
            checkpoint_path = download_pretrained(...)   # network!
        elif os.path.exists(pretrained):
            checkpoint_path = pretrained                 # local, no network

We additionally install a hard network guard around model construction so an
accidental download attempt fails loudly instead of silently pulling weights.

SCOPE GUARANTEES
    * k_shot is forced to 0 (zero-shot). Few-shot code paths are unreachable.
    * No training / fine-tuning / memory bank.
    * Prompts come from config and are versioned.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class MissingWeightsError(RuntimeError):
    """Raised when the local backbone weight file is absent.

    We deliberately STOP here and report the exact missing file instead of
    downloading it, per the project's manual-download policy.
    """


class NetworkBlockedError(RuntimeError):
    """Raised if any download is attempted during model construction."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml resolved against the project root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def sha256_of(path: Path, prefix_bytes: int | None = None) -> str:
    """SHA256 hex of a file (optionally only the first N bytes, for speed)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        if prefix_bytes is None:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(fh.read(prefix_bytes))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Prompt ensemble (config-driven, versioned)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptConfig:
    """Resolved prompt ensemble descriptor."""

    version: str
    class_name: str
    normal_states: list[str]
    anomalous_states: list[str]
    templates: list[str]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PromptConfig":
        p = cfg["prompts"]
        pc = cls(
            version=str(p["version"]),
            class_name=str(p["class_name"]),
            normal_states=[str(s) for s in p["normal_states"]],
            anomalous_states=[str(s) for s in p["anomalous_states"]],
            templates=[str(s) for s in p["templates"]],
        )
        if not pc.normal_states or not pc.anomalous_states or not pc.templates:
            msg = "prompt config incomplete: normal_states / anomalous_states / templates must all be non-empty"
            raise ValueError(msg)
        return pc

    def build(self) -> tuple[list[str], list[str]]:
        """Same combinatorial rule as anomalib's create_prompt_ensemble."""
        normal = [t.format(s) for s in (st.format(self.class_name) for st in self.normal_states) for t in self.templates]
        anomalous = [
            t.format(s) for s in (st.format(self.class_name) for st in self.anomalous_states) for t in self.templates
        ]
        return normal, anomalous

    @property
    def n_normal(self) -> int:
        return len(self.normal_states) * len(self.templates)

    @property
    def n_anomalous(self) -> int:
        return len(self.anomalous_states) * len(self.templates)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ZeroShotWinCLIP:
    """Zero-shot WinCLIP detector. Load once, reused for every image."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()
        m = self.cfg["model"]

        if int(m.get("k_shot", 0)) != 0:
            msg = f"k_shot must be 0 (zero-shot only). Got {m['k_shot']}."
            raise ValueError(msg)

        self.device = torch.device(m.get("device", "cpu"))
        self.input_size = tuple(m.get("input_size", (240, 240)))
        self.prompts = PromptConfig.from_config(self.cfg)

        self.weights_path = (PROJECT_ROOT / m["weights_path"]).resolve()
        self._assert_weights_present()

        self.load_seconds: float = 0.0
        self._build_model()

        self.transform = Compose(
            [
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Resize(self.input_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    # -- weights -----------------------------------------------------------
    def _assert_weights_present(self) -> None:
        if not self.weights_path.is_file():
            msg = (
                "Backbone weights NOT FOUND - refusing to download.\n"
                f"  expected file : {self.weights_path}\n"
                "  expected name : vit_b_16_plus_240-laion400m_e31-8fb26589.pt\n"
                "  known SHA256  : 8fb26589...  (open_clip internal check)\n"
                "Place the file manually per DOWNLOAD_CHECKLIST.md, then re-run."
            )
            raise MissingWeightsError(msg)
        if self.weights_path.stat().st_size == 0:
            msg = f"Backbone weight file is empty (0 bytes): {self.weights_path}"
            raise MissingWeightsError(msg)

    # -- construction ------------------------------------------------------
    def _build_model(self) -> None:
        from anomalib.models.image.winclip import torch_model as winclip_torch

        if self.prompts.version == "v1-official-default" or True:
            # Replace the module-level prompt factory with our config-driven one.
            # This is the ONLY place prompts enter the model.
            normal, anomalous = self.prompts.build()
            winclip_torch.create_prompt_ensemble = lambda _class_name=None, _n=normal, _a=anomalous: (_n, _a)

        # Redirect the pretrained tag to the local file (no network possible).
        winclip_torch.PRETRAINED = str(self.weights_path)

        t0 = time.perf_counter()
        with _no_downloads():
            self.model = winclip_torch.WinClipModel(
                class_name=self.prompts.class_name,
                reference_images=None,  # zero-shot: no reference/memory images
                scales=tuple(self.cfg["model"].get("scales", (2, 3))),
                apply_transform=False,
            )
        self.model.eval()
        self.model.to(self.device)
        self.load_seconds = time.perf_counter() - t0

        self._verify_zero_shot()

    def _verify_zero_shot(self) -> None:
        if int(getattr(self.model, "k_shot", -1)) != 0:
            msg = f"model unexpectedly in few-shot mode (k_shot={self.model.k_shot})"
            raise RuntimeError(msg)
        te = self.model.text_embeddings
        if tuple(te.shape) != (2, 640):
            msg = f"unexpected text embedding shape {tuple(te.shape)}, expected (2, 640)"
            raise RuntimeError(msg)

    # -- inference ---------------------------------------------------------
    def prepare(self, image: Image.Image) -> torch.Tensor:
        """PIL -> normalized tensor of shape (1, 3, H, W)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_tensor(self, batch: torch.Tensor) -> tuple[float, np.ndarray]:
        """Return (image_score, anomaly_map) for a prepared batch of one image."""
        out = self.model(batch)
        score = float(out.pred_score.reshape(-1)[0].item())
        amap = out.anomaly_map[0].detach().cpu().numpy().astype(np.float32)
        return score, amap

    def predict_path(self, image_path: str | Path) -> dict[str, Any]:
        """Full single-image inference returning score, map and timing."""
        image_path = Path(image_path)
        t_read = time.perf_counter()
        with Image.open(image_path) as im:
            pil = im.convert("RGB")
            size = pil.size
        tensor = self.prepare(pil)
        t_infer = time.perf_counter()
        score, amap = self.predict_tensor(tensor)
        infer_ms = (time.perf_counter() - t_infer) * 1000.0
        total_ms = (time.perf_counter() - t_read) * 1000.0
        return {
            "image_path": str(image_path),
            "image_size": list(size),
            "score": score,
            "anomaly_map": amap,
            "infer_ms": round(infer_ms, 2),
            "total_ms": round(total_ms, 2),
        }

    # -- metadata ----------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        return {
            "backbone": self.cfg["model"]["backbone"],
            "weights_path": str(self.weights_path),
            "weights_sha256": sha256_of(self.weights_path),
            "weights_bytes": self.weights_path.stat().st_size,
            "temperature": float(self.cfg["model"]["temperature"]),
            "scales": list(self.cfg["model"].get("scales", (2, 3))),
            "input_size": list(self.input_size),
            "k_shot": 0,
            "device": str(self.device),
            "prompt_version": self.prompts.version,
            "class_name": self.prompts.class_name,
            "n_normal_prompts": self.prompts.n_normal,
            "n_anomalous_prompts": self.prompts.n_anomalous,
            "load_seconds": round(self.load_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Network guard
# ---------------------------------------------------------------------------
class _no_downloads:
    """Context manager that turns any outbound HTTP download into an error.

    We only veto the module-level download helpers used by open_clip;
    nothing legitimate in model construction needs the network.
    """

    def __enter__(self):
        import open_clip.pretrained as ocp

        self._patches = []
        for name in ("download_pretrained_from_url", "download_pretrained_from_hf", "download_pretrained"):
            orig = getattr(ocp, name, None)
            if orig is None:
                continue

            def _boom(*a, __name=name, **k):
                msg = (
                    f"Blocked a pretrained-weights download attempt via open_clip.pretrained.{__name}. "
                    "This project forbids automatic weight downloads."
                )
                raise NetworkBlockedError(msg)

            setattr(ocp, name, _boom)
            self._patches.append((name, orig))
        return self

    def __exit__(self, *exc):
        import open_clip.pretrained as ocp

        for name, orig in self._patches:
            setattr(ocp, name, orig)
        return False


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------
def decide(score: float, threshold: float) -> str:
    """OK / NG decision. NG when score >= threshold."""
    return "NG" if score >= threshold else "OK"


def load_threshold(cfg: dict[str, Any] | None = None) -> float:
    cfg = cfg or load_config()
    val = cfg.get("threshold", {}).get("value")
    if val is None:
        msg = (
            "Threshold is not calibrated yet. Run:\n"
            "  python scripts/calibrate_threshold.py\n"
            "It derives the threshold from test/good scores only "
            "(mean + k*std) and writes it back into config.yaml."
        )
        raise RuntimeError(msg)
    return float(val)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _normalise_map(amap: np.ndarray, threshold: float | None) -> tuple[np.ndarray, float, float]:
    """Map raw anomaly scores to 0..1 for display. Returns (norm, v_lo, v_hi).

    IMPORTANT: the window is derived from the MAP'S OWN distribution, not from
    the decision threshold. The threshold is an IMAGE-level quantity; the map's
    attention mass is usually well below it, so anchoring the window on the
    threshold drove every normalized value into a narrow low band and the
    overlay came out flat. The threshold is still reported in the UI text; it
    does not need to be a colour position on the map.
    """
    del threshold  # kept in the signature for call-site stability
    lo = float(np.percentile(amap, 2.0))
    hi = float(np.percentile(amap, 99.5))
    if hi <= lo:
        hi = float(amap.max())
        lo = float(amap.min())
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((amap - lo) / (hi - lo), 0.0, 1.0)
    # Gamma > 1 pushes the broad, low-level response down to near-black so only
    # the genuinely hottest region lights up. (Gamma < 1 made everything glow,
    # which is unreadable: an OK part looked as "hot" as a defective one.)
    return np.power(norm, 2.2), lo, hi


def render_overlay(
    image_path: str | Path,
    anomaly_map: np.ndarray,
    out_path: str | Path,
    alpha: float = 0.75,
    threshold: float | None = None,
) -> None:
    """Save a BARE anomaly overlay PNG: same size as the input, no axes,
    no colorbar, no titles.

    This is what the web UI displays next to the untouched original. All
    numbers (score / threshold / verdict) live in the HTML around it, so the
    image itself must carry nothing but the picture - otherwise it gets
    letterboxed into a square and becomes unreadable.
    """
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

    h, w = rgb.shape[:2]
    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.shape != (h, w):
        amap = np.asarray(
            Image.fromarray(amap, mode="F").resize((w, h), resample=Image.BILINEAR),
            dtype=np.float32,
        )

    norm, _, _ = _normalise_map(amap, threshold)

    import matplotlib

    colormap = matplotlib.colormaps["inferno"]
    heat = colormap(norm)[..., :3]

    # The blend weight follows norm itself (already gamma-compressed), so the
    # map is ALWAYS visible while the cool majority of the image keeps the
    # original photo. A fixed cut-off would be unreachable for narrow maps and
    # silently flatten the overlay into a plain photo.
    # No floor: pixels at the map minimum stay EXACTLY the original photo, so a
    # normal part reads as "nothing found" instead of a global orange wash.
    weight = norm
    blend = alpha * weight[..., None]
    blended = np.clip((1 - blend) * rgb + blend * heat, 0, 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((blended * 255).round().astype(np.uint8)).save(out_path)


def overlay_heatmap(
    image_path: str | Path,
    anomaly_map: np.ndarray,
    out_path: str | Path,
    alpha: float = 0.55,
    score: float | None = None,
    threshold: float | None = None,
    decision: str | None = None,
) -> None:
    """Save a side-by-side [original | anomaly overlay] PNG for REPORTS/CLI.

    DESIGN NOTES (why it looks like this)
    -------------------------------------
    WinCLIP's anomaly map is numerically narrow: typical per-image spread is
    only ~0.03-0.05 while the absolute values sit around 0.42-0.48. A colour
    scale anchored on the *decision threshold* therefore compresses the entire
    map into a few hundredths of the colormap and renders it as one flat tint.

    The map's job here is LOCALISATION ("where is it hotter?"), not absolute
    calibration - the absolute number is already shown in the score, the
    threshold bar and the JSON. So:
      * colours are normalised PER IMAGE (min..max), plus a robust stretch
        so a single hot pixel cannot wash out the rest;
      * the colormap is perceptually uniform 'inferno' (dark = normal,
        bright yellow = most anomalous). We deliberately avoid a diverging
        blue/red scheme: a blue region would read as "healthy" on an image
        that is in fact defective;
      * a focus mask keeps the top anomalies saturated while the rest of the
        image stays close to the original photo, so the defect pops out;
      * the colorbar is labelled with the REAL score values at the ticks,
        so the viewer can still map colour back to numbers.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

    h, w = rgb.shape[:2]
    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.shape != (h, w):
        amap = np.asarray(
            Image.fromarray(amap, mode="F").resize((w, h), resample=Image.BILINEAR),
            dtype=np.float32,
        )

    lo, hi = float(amap.min()), float(amap.max())

    # --- normalisation (shared with render_overlay) ------------------------
    # Two competing goals:
    #   (a) the hot spot must be VISIBLE (the map's spread is only ~0.03);
    #   (b) a NORMAL part must not look alarming, otherwise every OK part
    #       would show a bright blob and the operator would stop trusting it.
    # The colour scale is anchored on the DECISION THRESHOLD while the window
    # width follows the image's own contrast, so the decision boundary is at a
    # meaningful colour position and structure is still visible.
    norm, v_lo, v_hi = _normalise_map(amap, threshold)
    if v_hi <= v_lo:
        v_lo, v_hi = lo, (hi if hi > lo else lo + 1e-6)

    colormap = matplotlib.colormaps["inferno"]
    heat = colormap(norm)[..., :3]

    # --- focus mask: weighted by the map itself, never thresholded ---------
    # (A fixed cut-off can be unreachable for narrow maps, which silently
    # flattens the overlay into a plain photo - see render_overlay notes.)
    weight = 0.06 + 0.94 * norm
    blend = alpha * weight[..., None]
    blended = np.clip((1 - blend) * rgb + blend * heat, 0, 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8), dpi=130)
    fig.patch.set_facecolor("#141a21")
    axes[0].imshow(rgb)
    axes[0].set_title("input", fontsize=10, color="#dbe6f0")

    axes[1].imshow(blended)
    title = "anomaly heatmap (red/yellow = hotter)"
    axes[1].set_title(title, fontsize=10, color="#dbe6f0")

    # Colorbar ticks shown with the TRUE anomaly-score values they correspond to.
    ticks = np.linspace(0, 1, 5)
    true_vals = v_lo + (ticks ** (1 / 2.2)) * (v_hi - v_lo)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.02, ticks=ticks)
    cbar.ax.set_yticklabels([f"{v:.4f}" for v in true_vals], fontsize=6.5, color="#dbe6f0")
    cbar.set_label("anomaly score (per-image scale)", fontsize=7, color="#dbe6f0")
    cbar.outline.set_edgecolor("#3a4757")
    cbar.ax.tick_params(colors="#dbe6f0")

    # A compact score/verdict strip underneath both panels.
    strip = ""
    if score is not None:
        strip += f"score = {score:.4f}"
    if threshold is not None:
        strip += f"    threshold = {threshold:.4f}"
    if decision is not None:
        strip += f"    verdict = {decision}"
    if strip:
        fig.text(0.5, 0.03, strip, ha="center", fontsize=9.5,
                 color="#ffd166" if decision == "NG" else "#7ee787", family="monospace")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def write_json(payload: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


__all__ = [
    "MissingWeightsError",
    "NetworkBlockedError",
    "PromptConfig",
    "ZeroShotWinCLIP",
    "decide",
    "load_config",
    "load_threshold",
    "overlay_heatmap",
    "render_overlay",
    "sha256_of",
    "write_json",
]
