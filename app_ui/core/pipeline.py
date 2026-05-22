from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import tifffile


# ── Make project root importable ─────────────────────────────────────────────
# app_ui/core/pipeline.py  -> parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Import your real backend ─────────────────────────────────────────────────
# These imports assume your current project structure, e.g.:
# Proyecto Final Astrocitos/
#   final_binary_model.py
#   temporal/pipeline_temp.py
try:
    import final_binary_model
except Exception as exc:
    raise ImportError(
        "No se pudo importar final_binary_model.py desde la raíz del proyecto."
    ) from exc

try:
    from temporal import pipeline_temp
except Exception as exc:
    raise ImportError(
        "No se pudo importar temporal/pipeline_temp.py desde la raíz del proyecto."
    ) from exc


# ── Data classes expected by the UI ──────────────────────────────────────────

@dataclass
class ClassifierResult:
    prediction: str                    # "Control" | "Pro-inflammatory"
    prob_control: float                # 0.0 – 1.0
    prob_proinflam: float              # 0.0 – 1.0
    median_thickness: float
    median_tortuosity: float
    median_segment_length: float
    all_features: dict[str, float] = field(default_factory=dict)
    wavelet_img: Optional[np.ndarray] = None
    mask_img: Optional[np.ndarray] = None
    skeleton_img: Optional[np.ndarray] = None


@dataclass
class ProgressionResult:
    inflammatory_score: float          # 0 = control-like, 100 = inflamed-like
    recovery_score: float              # 100 - inflammatory_score
    median_thickness: float
    median_segment_length: float
    all_features: dict[str, float] = field(default_factory=dict)
    wavelet_img: Optional[np.ndarray] = None
    mask_img: Optional[np.ndarray] = None
    skeleton_img: Optional[np.ndarray] = None
    n_ctrl_refs: int = 0
    n_proinf_refs: int = 0
    mode: str = "anchored"             # "anchored" | "absolute"
    note: str = ""
    warning: str = ""


# ── Small helpers ─────────────────────────────────────────────────────────────

def _read_tif(path: str) -> np.ndarray:
    """Read a TIFF image and return it as numpy array."""
    return tifffile.imread(path)


def _to_float_display(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Convert arrays to float32 for UI display widgets.
    Keeps shape as-is.
    """
    if arr is None:
        return None
    return np.asarray(arr).astype(np.float32)


def _normalize_prediction_label(label: str) -> str:
    """
    Map backend binary labels to UI labels.
    """
    if label in {"CTRL", "Control", "control"}:
        return "Control"
    return "Pro-inflammatory"


def _safe_get_feature(features: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(features.get(key, default))
    except Exception:
        return float(default)


# ── Public API used by the UI ────────────────────────────────────────────────

def run_classifier(image_path: str) -> ClassifierResult:
    """
    Binary classification wrapper.

    Expected backend:
      final_binary_model.predict_inflammatory_state(img)

    Expected result keys from your backend:
      pred_label
      prob_CTRL
      prob_72hs_LPS
      model_features
      optional: all_features, wavelet, mask, skeleton
    """
    img = _read_tif(image_path)

    result = final_binary_model.predict_inflammatory_state(img)

    pred_label = result.get("pred_label", "CTRL")
    prediction = _normalize_prediction_label(pred_label)

    model_features = result.get("model_features", {})
    all_features = result.get("all_features", model_features)

    prob_ctrl = float(result.get("prob_CTRL", 0.0))
    prob_pi = float(result.get("prob_72hs_LPS", 1.0 - prob_ctrl))

    return ClassifierResult(
        prediction=prediction,
        prob_control=prob_ctrl,
        prob_proinflam=prob_pi,
        median_thickness=_safe_get_feature(model_features, "median_thickness"),
        median_tortuosity=_safe_get_feature(model_features, "median_tortuosity"),
        median_segment_length=_safe_get_feature(model_features, "median_segment_length"),
        all_features=all_features,
        wavelet_img=_to_float_display(result.get("wavelet")),
        mask_img=_to_float_display(result.get("mask")),
        skeleton_img=_to_float_display(result.get("skeleton")),
    )


def compute_anchors(
    ctrl_paths: list[str],
    proinf_paths: list[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """
    Build multi-reference anchors for the anchored temporal mode.

    The UI template already calls this function first, then passes the returned
    objects into run_progression_anchored(). So instead of returning only
    averaged features, we return a small anchor package containing:
      - the loaded images
      - n_images
      - the averaged reference features

    That lets the UI keep its existing flow unchanged.
    """
    if not ctrl_paths:
        raise ValueError("No se cargaron imágenes control.")
    if not proinf_paths:
        raise ValueError("No se cargaron imágenes proinflamatorias.")

    ctrl_images = [_read_tif(p) for p in ctrl_paths]
    proinf_images = [_read_tif(p) for p in proinf_paths]

    ctrl_anchor = pipeline_temp.compute_reference_anchor_from_images(ctrl_images)
    proinf_anchor = pipeline_temp.compute_reference_anchor_from_images(proinf_images)

    ctrl_pkg = {
        "images": ctrl_images,
        "n_images": len(ctrl_images),
        "median_thickness": float(ctrl_anchor["median_thickness"]),
        "median_segment_length": float(ctrl_anchor["median_segment_length"]),
    }

    proinf_pkg = {
        "images": proinf_images,
        "n_images": len(proinf_images),
        "median_thickness": float(proinf_anchor["median_thickness"]),
        "median_segment_length": float(proinf_anchor["median_segment_length"]),
    }

    return ctrl_pkg, proinf_pkg


def run_progression_anchored(
    image_path: str,
    ctrl_anchor: dict[str, object],
    proinf_anchor: dict[str, object],
) -> ProgressionResult:
    """
    Anchored temporal progression wrapper.

    Uses your FINAL relative multi-reference mode:
      pipeline_temp.predict_temporal_progression_score_anchored(
          img_new,
          ctrl_images,
          inflam_images
      )
    """
    img = _read_tif(image_path)

    ctrl_images = ctrl_anchor.get("images", [])
    proinf_images = proinf_anchor.get("images", [])

    if not ctrl_images:
        raise ValueError("El anchor control no contiene imágenes.")
    if not proinf_images:
        raise ValueError("El anchor proinflamatorio no contiene imágenes.")

    result = pipeline_temp.predict_temporal_progression_score_anchored(
        img_new=img,
        ctrl_images=ctrl_images,
        inflam_images=proinf_images,
    )

    score_features = result.get("score_features", {})
    all_features = result.get("all_features", score_features)

    return ProgressionResult(
        inflammatory_score=float(result.get("inflammatory_score", 0.0)),
        recovery_score=float(result.get("recovery_score", 100.0)),
        median_thickness=_safe_get_feature(score_features, "median_thickness"),
        median_segment_length=_safe_get_feature(score_features, "median_segment_length"),
        all_features=all_features,
        wavelet_img=_to_float_display(result.get("wavelet")),
        mask_img=_to_float_display(result.get("mask")),
        skeleton_img=_to_float_display(result.get("skeleton")),
        n_ctrl_refs=int(ctrl_anchor.get("n_images", len(ctrl_images))),
        n_proinf_refs=int(proinf_anchor.get("n_images", len(proinf_images))),
        mode="anchored",
        note=result.get(
            "note",
            (
                f"Score calculado usando {len(ctrl_images)} imágenes control y "
                f"{len(proinf_images)} imágenes inflamadas del mismo experimento."
            ),
        ),
        warning="",
    )


def run_progression_absolute(image_path: str) -> ProgressionResult:
    """
    Reference-free temporal progression wrapper.

    Uses your FINAL absolute temporal mode:
      pipeline_temp.predict_temporal_progression_score(img)
    """
    img = _read_tif(image_path)

    result = pipeline_temp.predict_temporal_progression_score(img)

    # Your current absolute backend may return "score" or "inflammatory_score".
    inflammatory_score = float(
        result["inflammatory_score"] if "inflammatory_score" in result else result["score"]
    )

    score_features = result.get("score_features", {})
    all_features = result.get("all_features", score_features)

    return ProgressionResult(
        inflammatory_score=inflammatory_score,
        recovery_score=100.0 - inflammatory_score,
        median_thickness=_safe_get_feature(score_features, "median_thickness"),
        median_segment_length=_safe_get_feature(score_features, "median_segment_length"),
        all_features=all_features,
        wavelet_img=_to_float_display(result.get("wavelet")),
        mask_img=_to_float_display(result.get("mask")),
        skeleton_img=_to_float_display(result.get("skeleton")),
        n_ctrl_refs=0,
        n_proinf_refs=0,
        mode="absolute",
        note="",
        warning=result.get(
            "warning",
            (
                "Score calculado sin referencias del experimento actual. "
                "La interpretación puede verse afectada por variabilidad biológica entre réplicas."
            ),
        ),
    )