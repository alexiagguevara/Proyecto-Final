from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "AstroMetrix"

def app_base_dir() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]

def resource_path(*parts: str) -> Path:
    return app_base_dir().joinpath(*parts)

def user_data_dir() -> Path:
    path = Path.home() / "Library" / "Application Support" / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path

def user_recent_dir() -> Path:
    path = user_data_dir() / "recent_assets"
    path.mkdir(parents=True, exist_ok=True)
    return path

def user_recent_json_path() -> Path:
    return user_data_dir() / "recent_analyses.json"