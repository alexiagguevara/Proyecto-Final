from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ASSETS_DIR = DATA_DIR / "recent_assets"
ASSETS_DIR.mkdir(exist_ok=True)

RECENT_PATH = DATA_DIR / "recent_analyses.json"
MAX_ITEMS = 12


def _read_all() -> list[dict]:
    if not RECENT_PATH.exists():
        return []
    try:
        return json.loads(RECENT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_all(items: list[dict]) -> None:
    RECENT_PATH.write_text(
        json.dumps(items, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def _delete_assets_for_entry(entry: dict) -> None:
    analysis_id = entry.get("id")
    if not analysis_id:
        return

    asset_dir = ASSETS_DIR / analysis_id
    if asset_dir.exists() and asset_dir.is_dir():
        shutil.rmtree(asset_dir, ignore_errors=True)


def _json_safe(obj: Any):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _array_to_pil(arr):
    if arr is None:
        return None

    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    if arr.ndim == 3:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    return None


def _save_asset_image(arr, out_path: Path) -> str | None:
    pil = _array_to_pil(arr)
    if pil is None:
        return None
    pil.save(out_path)
    return str(out_path.relative_to(BASE_DIR))


def save_recent_analysis(
    image_name: str,
    mode_key: str,
    mode_label: str,
    result_label: str,
    result_kind: str,
    snapshot: dict,
) -> str:
    items = _read_all()

    analysis_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{mode_key}"
    asset_dir = ASSETS_DIR / analysis_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    wavelet_path = _save_asset_image(snapshot.pop("wavelet_img", None), asset_dir / "wavelet.png")
    mask_path = _save_asset_image(snapshot.pop("mask_img", None), asset_dir / "mask.png")
    skeleton_path = _save_asset_image(snapshot.pop("skeleton_img", None), asset_dir / "skeleton.png")

    entry = {
        "id": analysis_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_name": image_name,
        "mode_key": mode_key,
        "mode_label": mode_label,
        "result_label": result_label,
        "result_kind": result_kind,
        "snapshot": _json_safe(snapshot),
        "assets": {
            "wavelet": wavelet_path,
            "mask": mask_path,
            "skeleton": skeleton_path,
        }
    }

    items.insert(0, entry)

    removed_items = items[MAX_ITEMS:]
    items = items[:MAX_ITEMS]

    for old_entry in removed_items:
        _delete_assets_for_entry(old_entry)

    _write_all(items)
    return analysis_id


def load_recent_analyses() -> list[dict]:
    return _read_all()


def load_recent_analysis_by_id(analysis_id: str) -> dict | None:
    items = _read_all()
    for item in items:
        if item.get("id") == analysis_id:
            return item
    return None


def load_asset_image(relative_path: str):
    if not relative_path:
        return None
    path = BASE_DIR / relative_path
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def clear_recent_analyses() -> None:
    items = _read_all()
    for entry in items:
        _delete_assets_for_entry(entry)

    if RECENT_PATH.exists():
        RECENT_PATH.unlink()

    ASSETS_DIR.mkdir(exist_ok=True)