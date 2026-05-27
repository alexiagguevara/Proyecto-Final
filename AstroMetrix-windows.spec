# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("customtkinter")

a = Analysis(
    ["app_ui/main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("final_binary_model.joblib", "."),
        ("final_binary_model_metadata.joblib", "."),
        ("temporal_score_metadata.joblib", "."),
        ("app_ui/assets", "app_ui/assets"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AstroMetrix",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="app_ui/assets/AstroMetrix.ico",
)