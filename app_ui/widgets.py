"""
widgets.py — Reusable custom widgets for AstroMetrix.

Widgets exposed:
    C               — colour palette dict
    Topbar          — top navigation bar with optional back button
    ModeCard        — clickable home-screen card
    UploadZone      — single-file .tif picker
    RefUploadZone   — multi-file .tif picker with live count
    ScoreBar        — segmented categorical score bar (tkinter Canvas)
    ProbBar         — horizontal probability bar
    FeatureCard     — mini card showing one feature value
    FeatureTable    — collapsible table of all extracted features
    CollapsibleSection — expandable section with arrow toggle
    ProcessingOutputs  — 3-panel wavelet / mask / skeleton preview
    Banner          — info / warning / success / danger banner
"""

from __future__ import annotations
import sys
import csv
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from typing import Callable, Optional
from pathlib import Path
from datetime import datetime


# ── Colour palette ────────────────────────────────────────────────────────────

C: dict[str, str] = {
    # greens
    "green_dark":   "#2D5A0E",
    "green_mid":    "#639922",
    "green_light":  "#86EFAC",
    # score ramp (used in ScoreBar segments)
    "seg_green":    "#3B6D11",
    "seg_amber":    "#BA7517",
    "seg_coral":    "#D85A30",
    "seg_red":      "#E24B4A",
    # semantic backgrounds
    "success_bg":   "#F0FDF4",
    "success_fg":   "#166534",
    "danger_bg":    "#FEF2F2",
    "danger_fg":    "#991B1B",
    "warning_bg":   "#FFFBEB",
    "warning_fg":   "#92400E",
    "info_bg":      "#EFF6FF",
    "info_fg":      "#1E40AF",
    # neutrals
    "text1":        "#111827",
    "text2":        "#6B7280",
    "text3":        "#9CA3AF",
    "border":       "#E5E7EB",
    "border_dark":  "#D1D5DB",
    "bg1":          "#FFFFFF",
    "bg2":          "#F9FAFB",
    "bg3":          "#F3F4F6",
    # fonts
    "mono":         "Courier New",
    "sans":         "Helvetica",
}
# -- Helpers --------------------------------------------------------------------
def make_responsive_wrap(label, parent, padding=40, min_wrap=220):
    """
    Hace que un CTkLabel ajuste su wraplength según el ancho disponible del parent.
    """
    def _update_wrap(_event=None):
        try:
            width = max(min_wrap, parent.winfo_width() - padding)
            label.configure(wraplength=width)
        except Exception:
            pass

    parent.bind("<Configure>", _update_wrap, add="+")
    _update_wrap()

CLICK_CURSOR = "pointinghand" if sys.platform == "darwin" else "hand2"

# ── Topbar ────────────────────────────────────────────────────────────────────

class Topbar(ctk.CTkFrame):
    """Top navigation bar."""

    def __init__(self, parent,
                 breadcrumb: str = "",
                 back_label: str = "",
                 back_cmd: Callable = None,
                 **kwargs):
        super().__init__(parent, fg_color=C["bg1"], corner_radius=0, **kwargs)

        # ── Top row ──────────────────────────────────────────────────────────
        row = ctk.CTkFrame(self, fg_color="transparent", height=52)
        row.pack(fill="x", padx=16, pady=(0, 0))
        row.pack_propagate(False)

        # Logo mark
        lm = ctk.CTkFrame(
            row,
            fg_color=C["success_bg"],
            corner_radius=6,
            width=28,
            height=28
        )
        lm.pack(side="left", padx=(0, 6), pady=12)
        lm.pack_propagate(False)

        ctk.CTkLabel(
            lm,
            text="✦",
            font=(C["sans"], 12),
            text_color=C["green_mid"]
        ).place(relx=.5, rely=.5, anchor="center")

        ctk.CTkLabel(
            row,
            text="AstroMetrix",
            font=(C["sans"], 15, "bold"),
            text_color=C["text1"]
        ).pack(side="left")

        if breadcrumb:
            ctk.CTkLabel(
                row,
                text=f"  /  {breadcrumb}",
                font=(C["mono"], 11),
                text_color=C["text3"]
            ).pack(side="left")

        if back_label and back_cmd:
            ctk.CTkButton(
                row,
                text=f"← {back_label}",
                command=back_cmd,
                fg_color="transparent",
                text_color=C["text2"],
                hover_color=C["bg2"],
                font=(C["sans"], 12),
                width=90,
                height=28,
                cursor=CLICK_CURSOR,
            ).pack(side="right")

        # ── Full-width divider ───────────────────────────────────────────────
        ctk.CTkFrame(
            self,
            height=1,
            fg_color=C["border"]
        ).pack(fill="x", padx=16, pady=(0, 0))


# ── Mode card ─────────────────────────────────────────────────────────────────

class ModeCard(ctk.CTkFrame):
    """Clickable mode-selection card for the home screen."""

    def __init__(self, parent, title: str, description: str,
                 icon_char: str, icon_bg: str, command: Callable, **kwargs):
        super().__init__(parent, fg_color=C["bg1"], corner_radius=10,
                         border_width=1, border_color=C["border"], **kwargs)
        self.configure(cursor=CLICK_CURSOR)
        self._cmd = command
        for w in (self,):
            w.bind("<Enter>",    lambda e: self.configure(fg_color=C["bg2"]))
            w.bind("<Leave>",    lambda e: self.configure(fg_color=C["bg1"]))
            w.bind("<Button-1>", lambda e: self._cmd())

        ico = ctk.CTkFrame(self, fg_color=icon_bg, corner_radius=8,
                           width=36, height=36)
        ico.configure(cursor=CLICK_CURSOR)
        ico.pack(anchor="w", padx=14, pady=(14, 8))
        ico.pack_propagate(False)
        ctk.CTkLabel(ico, text=icon_char, font=(C["sans"], 16),
                     text_color=C["text1"], cursor=CLICK_CURSOR).place(relx=.5, rely=.5, anchor="center")

        ctk.CTkLabel(self, text=title, font=(C["sans"], 13, "bold"),
                     text_color=C["text1"], anchor="w", cursor=CLICK_CURSOR).pack(anchor="w", padx=14)
        ctk.CTkLabel(self, text=description, font=(C["sans"], 11),
                     text_color=C["text2"], anchor="w",
                     justify="left", wraplength=180, cursor=CLICK_CURSOR).pack(
            anchor="w", padx=14, pady=(2, 14))


# ── Upload zones ──────────────────────────────────────────────────────────────

class UploadZone(ctk.CTkFrame):
    """Single-file .tif upload zone (click-to-browse) with two states:
    - empty: shows the upload prompt
    - selected: shows selected filename + Remove button
    """

    def __init__(self, parent,
                 label: str = "Drop .tif image here",
                 icon_color: str = None,
                 on_file: Callable[[str], None] = None,
                 on_clear: Callable[[], None] = None,
                 **kwargs):
        super().__init__(parent, fg_color=C["bg2"], corner_radius=8,
                         border_width=1, border_color=C["border"], **kwargs)

        self._on_file = on_file
        self._on_clear = on_clear
        self._icol = icon_color or C["success_fg"]
        self._label_text = label
        self._path: str | None = None

        self.configure(cursor=CLICK_CURSOR)
        self._build_empty()

    def _clear_children(self):
        for w in self.winfo_children():
            w.destroy()

    def _hover_on(self, _event=None):
        self.configure(border_color=C["green_mid"])

    def _hover_off(self, _event=None):
        self.configure(border_color=C["border"])

    def _build_empty(self):
        self._clear_children()

        self.configure(border_color=C["border"])
        self.bind("<Enter>", self._hover_on)
        self.bind("<Leave>", self._hover_off)
        self.bind("<Button-1>", lambda e: self._browse())

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.configure(cursor=CLICK_CURSOR)
        inner.pack(pady=20, padx=16, fill="x")
        inner.bind("<Enter>", self._hover_on)
        inner.bind("<Leave>", self._hover_off)
        inner.bind("<Button-1>", lambda e: self._browse())

        icon = ctk.CTkLabel(
            inner,
            text="↑",
            font=(C["sans"], 22, "bold"),
            text_color=self._icol,
            cursor=CLICK_CURSOR
        )
        icon.pack()
        icon.bind("<Enter>", self._hover_on)
        icon.bind("<Leave>", self._hover_off)
        icon.bind("<Button-1>", lambda e: self._browse())

        lbl = ctk.CTkLabel(
            inner,
            text=self._label_text,
            font=(C["sans"], 13, "bold"),
            text_color=C["text1"],
            cursor=CLICK_CURSOR
        )
        lbl.pack()
        lbl.bind("<Enter>", self._hover_on)
        lbl.bind("<Leave>", self._hover_off)
        lbl.bind("<Button-1>", lambda e: self._browse())

        sub = ctk.CTkLabel(
            inner,
            text="click to browse · .tif only",
            font=(C["mono"], 10),
            text_color=C["text3"],
            cursor=CLICK_CURSOR
        )
        sub.pack()
        sub.bind("<Enter>", self._hover_on)
        sub.bind("<Leave>", self._hover_off)
        sub.bind("<Button-1>", lambda e: self._browse())

    def _build_selected(self):
        self._clear_children()
        from pathlib import Path

        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(
            outer,
            text="Selected image",
            font=(C["mono"], 10, "bold"),
            text_color=C["text3"],
        ).pack(anchor="w", pady=(0, 6))

        row = ctk.CTkFrame(outer, fg_color="transparent")
        row.pack(fill="x")

        ctk.CTkLabel(
            row,
            text=Path(self._path).name if self._path else "",
            font=(C["mono"], 11),
            text_color=C["text1"],
            anchor="w",
        ).pack(side="left")

        ctk.CTkButton(
            row,
            text="Remove",
            command=self.clear,
            fg_color=C["danger_bg"],
            text_color=C["danger_fg"],
            hover_color=C["bg3"],
            height=28,
            width=80,
            font=(C["sans"], 11),
            cursor=CLICK_CURSOR,
        ).pack(side="right")

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select .tif image",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return

        self._path = path
        self._build_selected()

        if self._on_file:
            self._on_file(path)

    def clear(self):
        self._path = None
        self._build_empty()
        if self._on_clear:
            self._on_clear()

    def get_path(self):
        return self._path


class RefUploadZone(ctk.CTkFrame):

    """
    Multi-file .tif upload zone with:
    - browse button
    - file list
    - remove button per file
    - live count
    """

    def __init__(self, parent, label: str,
                 on_files: Callable[[list[str]], None] = None,
                 on_change: Callable[[list[str]], None] = None,
                 **kwargs):
        super().__init__(parent, fg_color=C["bg2"], corner_radius=8,
                         border_width=1, border_color=C["border"], **kwargs)

        self._on_files = on_files
        self._on_change = on_change
        self._paths: list[str] = []

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(header, text=label, font=(C["sans"], 12, "bold"),
                     text_color=C["text1"]).pack(side="left")

        self._count_lbl = ctk.CTkLabel(header, text="",
                                       font=(C["mono"], 11),
                                       text_color=C["success_fg"])
        self._count_lbl.pack(side="right")

        self._browse_btn = ctk.CTkButton(
            self,
            text="Browse .tif files",
            command=self._browse,
            fg_color=C["success_bg"],
            text_color=C["success_fg"],
            hover_color=C["bg3"],
            font=(C["sans"], 12),
            height=28,
            cursor=CLICK_CURSOR,
        )
        self._browse_btn.pack(anchor="w", padx=12, pady=(0, 8))

        self._list_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._list_frame.pack(fill="x", padx=12, pady=(0, 10))
        self._list_frame.pack_forget()

    def _browse(self):
        paths = list(filedialog.askopenfilenames(
            title="Select .tif reference images",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        ))
        if not paths:
            return

        new_paths = [p for p in paths if p not in self._paths]
        if not new_paths:
            return

        self._paths.extend(new_paths)
        self._refresh_list()

        if self._on_files:
            self._on_files(new_paths)

        if self._on_change:
            self._on_change(list(self._paths))

    def _refresh_list(self):
        from pathlib import Path

        for w in self._list_frame.winfo_children():
            w.destroy()

        n = len(self._paths)
        self._count_lbl.configure(
            text=f"✓  {n} image{'s' if n != 1 else ''} loaded" if n else ""
        )

        if n == 0:
            self._list_frame.pack_forget()
            return

        self._list_frame.pack(fill="x", padx=12, pady=(0, 10))

        for p in self._paths:
            row = ctk.CTkFrame(self._list_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)

            ctk.CTkLabel(
                row,
                text=Path(p).name,
                font=(C["mono"], 10),
                text_color=C["text1"],
                anchor="w",
            ).pack(side="left")

            ctk.CTkButton(
                row,
                text="Remove",
                command=lambda path=p: self.remove_path(path),
                fg_color=C["danger_bg"],
                text_color=C["danger_fg"],
                hover_color=C["bg3"],
                height=24,
                width=70,
                font=(C["sans"], 11),
                cursor=CLICK_CURSOR,
            ).pack(side="right")

    def remove_path(self, path: str):
        if path in self._paths:
            self._paths.remove(path)
            self._refresh_list()
            if self._on_change:
                self._on_change(list(self._paths))

    def get_paths(self) -> list[str]:
        return list(self._paths)

    def reset(self):
        self._paths = []
        self._refresh_list()
        if self._on_change:
            self._on_change(list(self._paths))


# ── Score bar (Canvas) ────────────────────────────────────────────────────────

class ScoreBar(tk.Canvas):
    """
    Segmented categorical score bar with threshold markers and a position needle.

    mode="inflammatory"  → green ▸ amber ▸ coral ▸ red  (0 = ctrl, 100 = inflamed)
    mode="recovery"      → red   ▸ coral ▸ amber ▸ green (0 = inflamed, 100 = ctrl)
    """

    _INFLAM_T  = [16.81, 48.56, 81.75]
    _RECOVERY_T = [18.25, 51.44, 83.19]
    _INFLAM_C  = ["#3B6D11", "#BA7517", "#D85A30", "#E24B4A"]
    _RECOVERY_C = ["#E24B4A", "#D85A30", "#BA7517", "#3B6D11"]

    def __init__(self, parent, score: float = 0.0,
                 mode: str = "inflammatory",
                 height: int = 46, **kwargs):
        kwargs["height"] = height
        kwargs["highlightthickness"] = 0
        kwargs["bd"] = 0
        kwargs.setdefault("bg", "#FFFFFF")
        super().__init__(parent, **kwargs)
        self.score = max(0.0, min(100.0, score))
        self.mode  = mode
        self.bind("<Configure>", self._draw)

    def set_score(self, score: float):
        self.score = max(0.0, min(100.0, score))
        self._draw()

    def _draw(self, _event=None):
        self.delete("all")
        W = self.winfo_width()
        H = self.winfo_height()
        if W < 20:
            return

        BAR_TOP = 14
        BAR_H   = 10
        R       = BAR_H // 2          # corner radius
        PAD     = 2                    # left/right padding

        thresholds = ([0] + self._INFLAM_T  + [100] if self.mode == "inflammatory"
                      else [0] + self._RECOVERY_T + [100])
        colors     = (self._INFLAM_C if self.mode == "inflammatory"
                      else self._RECOVERY_C)

        effective_w = W - 2 * PAD

        def tx(pct):
            return PAD + int(pct / 100 * effective_w)

        # Draw segments
        for i in range(4):
            x0 = tx(thresholds[i])
            x1 = tx(thresholds[i + 1])
            self.create_rectangle(x0, BAR_TOP, x1, BAR_TOP + BAR_H,
                                  fill=colors[i], outline="")

        # Rounded end caps
        self.create_oval(PAD, BAR_TOP,
                         PAD + BAR_H, BAR_TOP + BAR_H,
                         fill=colors[0], outline="")
        self.create_oval(W - PAD - BAR_H, BAR_TOP,
                         W - PAD,          BAR_TOP + BAR_H,
                         fill=colors[-1], outline="")

        # Threshold ticks & labels
        for t in thresholds[1:-1]:
            x = tx(t)
            self.create_line(x, BAR_TOP - 4, x, BAR_TOP,
                             fill="#9CA3AF", width=1)
            self.create_text(x, BAR_TOP - 5,
                             text=f"{t:.0f}",
                             font=("Courier New", 7),
                             fill="#9CA3AF", anchor="s")

        # Needle
        mx = tx(self.score)
        mx = max(PAD + 2, min(W - PAD - 2, mx))
        self.create_rectangle(mx - 1, BAR_TOP - 3,
                              mx + 1, BAR_TOP + BAR_H + 3,
                              fill="#1F2937", outline="")


# ── Probability bar ───────────────────────────────────────────────────────────

class ProbBar(ctk.CTkFrame):
    """Single probability bar row: label — track — percentage."""

    def __init__(self, parent, label: str, value: float,
                 color: str, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(row, text=label, font=(C["sans"], 11),
                     text_color=C["text3"]).pack(side="left")
        ctk.CTkLabel(row, text=f"{value:.0f}%", font=(C["mono"], 11),
                     text_color=C["text3"]).pack(side="right")

        track = ctk.CTkFrame(self, height=6, corner_radius=3,
                             fg_color=C["bg3"])
        track.pack(fill="x", pady=(2, 4))
        fill = ctk.CTkFrame(track, height=6, corner_radius=3,
                            fg_color=color)
        fill.place(relwidth=max(0.02, value / 100), relheight=1)


# ── Feature card ──────────────────────────────────────────────────────────────

class FeatureCard(ctk.CTkFrame):
    """Mini card displaying one feature name + value."""

    def __init__(self, parent, label: str, value: str, **kwargs):
        super().__init__(parent, fg_color=C["bg2"], corner_radius=8, **kwargs)
        ctk.CTkLabel(self, text=label, font=(C["mono"], 9),
                     text_color=C["text3"], anchor="w").pack(
            anchor="w", padx=10, pady=(8, 0))
        ctk.CTkLabel(self, text=value, font=(C["sans"], 17, "bold"),
                     text_color=C["text1"], anchor="w").pack(
            anchor="w", padx=10, pady=(0, 8))


# ── Collapsible section ───────────────────────────────────────────────────────

class CollapsibleSection(ctk.CTkFrame):
    """Frame with a header that toggles an expandable body."""

    def __init__(self, parent, title: str, start_open: bool = False, **kwargs):
        super().__init__(parent, fg_color="transparent",
                         corner_radius=0, **kwargs)

        self._open = start_open

        # Header button
        self._header = ctk.CTkFrame(self, fg_color=C["bg2"],
                                    corner_radius=8,
                                    border_width=1,
                                    border_color=C["border"],
                                    height=40)
        self.configure(cursor=CLICK_CURSOR)
        self._header.pack(fill="x")
        self._header.pack_propagate(False)
        self._header.bind("<Button-1>", self._toggle)

        self._title_lbl = ctk.CTkLabel(
            self._header, text=title,
            font=(C["sans"], 13, "bold"), text_color=C["text1"], cursor=CLICK_CURSOR)
        self._title_lbl.pack(side="left", padx=14)
        self._title_lbl.bind("<Button-1>", self._toggle)

        self._arrow = ctk.CTkLabel(
            self._header, text="▶",
            font=(C["sans"], 10), text_color=C["text3"], width=24, cursor=CLICK_CURSOR)
        self._arrow.pack(side="right", padx=14)
        self._arrow.bind("<Button-1>", self._toggle)

        # Body (hidden by default)
        self._body = ctk.CTkFrame(self, fg_color=C["bg1"],
                                  corner_radius=8,
                                  border_width=1,
                                  border_color=C["border"])

        self._set_open(self._open)

    def _set_open(self, is_open: bool):
        self._open = is_open
        if self._open:
            self.body.pack(fill="x", pady=(6, 0))
            self._arrow.configure(text="▼")
        else:
            self.body.pack_forget()
            self._arrow.configure(text="▶")

    def _toggle(self, _event=None):
        self._set_open(not self._open)

    @property
    def body(self) -> ctk.CTkFrame:
        return self._body


# ── Feature table (inside CollapsibleSection) ─────────────────────────────────

class FeatureTable(CollapsibleSection):
    """
    Collapsible table showing all extracted features.
    Pass features as {name: value} dict.
    Shows first N_PREVIEW rows; expand to see all.
    """
    N_PREVIEW = 6

    def __init__(self, parent, features: dict[str, float] = None, **kwargs):
        super().__init__(parent, title="All extracted features", start_open=False, **kwargs)
        self._features = features or {}
        self._rows_extra: list[ctk.CTkFrame] = []
        self._expanded = False
        self._build_download_button()
        self._build_table()

    def _build_download_button(self):
        self._download_btn = ctk.CTkButton(
            self._header,
            text="⤓",
            command=self._download_features_csv,
            fg_color="transparent",
            text_color=C["text2"],
            hover_color=C["bg3"],
            font=(C["sans"], 14, "bold"),
            width=28,
            height=26,
            cursor=CLICK_CURSOR,
        )
        self._download_btn.pack(side="right", padx=(0, 6))

        # mover flecha un poco más a la derecha si ya existe
        try:
            self._arrow.pack_forget()
            self._arrow.pack(side="right", padx=14)
        except Exception:
            pass
    
    def set_features(self, features: dict[str, float]):
        self._features = features
        # Rebuild
        for w in self.body.winfo_children():
            w.destroy()
        self._rows_extra = []
        self._expanded = False
        self._build_table()

    def _build_table(self):
        items = list(self._features.items())

        grid = ctk.CTkFrame(self.body, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=8)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        for i, (k, v) in enumerate(items):
            row = ctk.CTkFrame(grid, fg_color="transparent", height=24)
            row.grid(row=i // 2, column=i % 2, sticky="ew", padx=4)

            ctk.CTkLabel(row, text=k, font=(C["mono"], 10),
                         text_color=C["text3"], anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=f"{v:.4g}" if isinstance(v, float) else str(v),
                         font=(C["mono"], 10),
                         text_color=C["text1"], anchor="e").pack(side="right")

            if i // 2 >= self.N_PREVIEW // 2:
                row.grid_remove()
                self._rows_extra.append(row)

        if self._rows_extra:
            self._toggle_btn = ctk.CTkButton(
                self.body, text=f"view all {len(items)} features ↓",
                command=self._toggle_extra,
                fg_color="transparent",
                text_color=C["info_fg"],
                hover_color=C["bg2"],
                font=(C["sans"], 11),
                height=28,
            )
            self._toggle_btn.pack(pady=(0, 6))

    def _toggle_extra(self):
        self._expanded = not self._expanded
        for row in self._rows_extra:
            if self._expanded:
                row.grid()
            else:
                row.grid_remove()
        self._toggle_btn.configure(
            text="collapse ↑" if self._expanded
            else f"view all {len(self._features)} features ↓"
        )

    def _download_features_csv(self):
        if not self._features:
            return

        path = filedialog.asksaveasfilename(
            title="Save extracted features",
            defaultextension=".csv",
            initialfile="extracted_features.csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Feature", "Value"])
            for k, v in self._features.items():
                writer.writerow([k, v])

# ── Expandable image ──────────────────────────────────────────────────

class ExpandableImageCard(ctk.CTkFrame):
    """
    Safer image card for macOS:
    - title
    - thumbnail
    - Open button
    - Download button
    - large viewer (non-fullscreen)
    """

    def __init__(self, parent, title: str, array: np.ndarray | None, **kwargs):
        super().__init__(parent, fg_color=C["bg2"], corner_radius=10, **kwargs)
        self._title = title
        self._array = array
        self._pil = self._array_to_pil(array) if array is not None else None
        self._thumb_ctk = None
        self._viewer_ctk = None

        self._build()

    def _array_to_pil(self, arr):
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

    def _build(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 6))

        ctk.CTkLabel(
            header,
            text=self._title,
            font=(C["sans"], 12, "bold"),
            text_color=C["text1"],
        ).pack(side="left")

        action_row = ctk.CTkFrame(header, fg_color="transparent")
        action_row.pack(side="right")

        ctk.CTkButton(
            action_row,
            text="⛶",
            command=self._open_viewer,
            fg_color="transparent",
            text_color=C["text2"],
            hover_color=C["bg3"],
            font=(C["sans"], 14),
            width=52,
            height=26,
            cursor=CLICK_CURSOR,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            action_row,
            text="⤓",
            command=self._download_image,
            fg_color="transparent",
            text_color=C["text2"],
            hover_color=C["bg3"],
            font=(C["sans"], 16),
            width=28,
            height=26,
            cursor=CLICK_CURSOR,
        ).pack(side="left")

        if self._pil is None:
            ctk.CTkLabel(
                self,
                text="No image",
                font=(C["sans"], 12),
                text_color=C["text3"],
            ).pack(expand=True, pady=20)
            return

        thumb = self._pil.copy()
        thumb.thumbnail((280, 210))
        self._thumb_ctk = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=thumb.size)

        img_lbl = ctk.CTkLabel(self, text="", image=self._thumb_ctk)
        img_lbl.pack(padx=10, pady=(0, 10))

    def _download_image(self):
        if self._pil is None:
            return

        default_name = self._title.lower().replace(" ", "_") + ".png"
        path = filedialog.asksaveasfilename(
            title="Save image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
        )
        if not path:
            return

        self._pil.save(path)

    def _open_viewer(self):
        if self._pil is None:
            return

        viewer = ctk.CTkToplevel(self)
        viewer.title(self._title)
        viewer.configure(fg_color=C["bg1"])

        sw = viewer.winfo_screenwidth()
        sh = viewer.winfo_screenheight()
        viewer.geometry(f"{int(sw * 0.8)}x{int(sh * 0.8)}+80+60")

        top = ctk.CTkFrame(viewer, fg_color="transparent")
        top.pack(fill="x", padx=18, pady=(14, 8))

        ctk.CTkLabel(
            top,
            text=self._title,
            font=(C["sans"], 18, "bold"),
            text_color=C["text1"],
        ).pack(side="left")

        ctk.CTkButton(
            top,
            text="Download ⤓",
            command=self._download_image,
            fg_color=C["bg2"],
            text_color=C["text1"],
            hover_color=C["bg3"],
            cursor=CLICK_CURSOR,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            top,
            text="Close",
            command=viewer.destroy,
            fg_color=C["danger_bg"],
            text_color=C["danger_fg"],
            hover_color=C["bg3"],
            cursor=CLICK_CURSOR,
        ).pack(side="right")

        body = ctk.CTkFrame(viewer, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        img = self._pil.copy()
        img.thumbnail((int(sw * 0.72), int(sh * 0.68)))
        self._viewer_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)

        lbl = ctk.CTkLabel(body, text="", image=self._viewer_ctk)
        lbl.pack(expand=True)

        viewer.bind("<Escape>", lambda e: viewer.destroy())


# ── Processing outputs strip ──────────────────────────────────────────────────

class ProcessingOutputs(ctk.CTkFrame):
    """ Collapsible section showing wavelet / mask / skeleton thumbnails. 
    Shows wavelet / mask / skeleton as expandable image cards."""

    def __init__(self, parent,
                 wavelet: np.ndarray | None,
                 mask: np.ndarray | None,
                 skeleton: np.ndarray | None,
                 **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        section = CollapsibleSection(self, title="Processing outputs", start_open=False)
        section.pack(fill="x")

        grid = ctk.CTkFrame(section.body, fg_color="transparent")
        grid.pack(fill="x", padx=4, pady=4)
        grid.columnconfigure((0, 1, 2), weight=1)

        ExpandableImageCard(grid, "Wavelet", wavelet).grid(
            row=0, column=0, padx=4, sticky="nsew"
        )
        ExpandableImageCard(grid, "Mask", mask).grid(
            row=0, column=1, padx=4, sticky="nsew"
        )
        ExpandableImageCard(grid, "Skeleton", skeleton).grid(
            row=0, column=2, padx=4, sticky="nsew"
        )

# ── Banner ────────────────────────────────────────────────────────────────────

_BANNER_STYLES = {
    "info":    (C["info_bg"],    C["info_fg"]),
    "warning": (C["warning_bg"], C["warning_fg"]),
    "success": (C["success_bg"], C["success_fg"]),
    "danger":  (C["danger_bg"],  C["danger_fg"]),
}


class Banner(ctk.CTkFrame):
    """Coloured info/warning/success/danger banner."""

    def __init__(self, parent, text: str, style: str = "info", **kwargs):
        bg, fg = _BANNER_STYLES.get(style, _BANNER_STYLES["info"])
        super().__init__(parent, fg_color=bg, corner_radius=8, **kwargs)
        self._label = ctk.CTkLabel(self, text=text,
                     font=(C["sans"], 12),
                     text_color=fg,
                     justify="left",
                     anchor="w",)
        self._label.pack(fill="x", padx=14, pady=10)
        make_responsive_wrap(self._label, self, padding=24, min_wrap=220)

# ── Recent analyses ────────────────────────────────────────────────────────────────────

def format_relative_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return ""

    delta = datetime.now() - dt
    seconds = int(delta.total_seconds())

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        mins = seconds // 60
        return f"{mins} min ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} h ago"
    if seconds < 172800:
        return "yesterday"
    days = seconds // 86400
    return f"{days} d ago"


def result_style(kind: str):
    mapping = {
        "Control": (C["success_bg"], C["success_fg"]),
        "Pro-inflammatory": (C["danger_bg"], C["danger_fg"]),

        "low": ("#EEF7EE", "#267A3E"),
        "mild": ("#F6F1DE", "#9A6B00"),
        "moderate": ("#F4EEDC", "#A44B00"),
        "high": (C["danger_bg"], C["danger_fg"]),

        "minimal_recovery": (C["danger_bg"], C["danger_fg"]),
        "partial_recovery": ("#F4EEDC", "#A44B00"),
        "substantial_recovery": ("#EEF7EE", "#267A3E"),
        "near_complete_recovery": (C["success_bg"], C["success_fg"]),
    }
    return mapping.get(kind, (C["bg2"], C["text2"]))


def mode_icon(mode_key: str) -> str:
    if mode_key == "classifier":
        return "◫"
    if mode_key == "progression_anchored":
        return "↗"
    if mode_key == "progression_free":
        return "↗"
    return "•"


def mode_style(mode_key: str):
    mapping = {
        "classifier": (C["success_bg"], C["success_fg"]),
        "progression_anchored": (C["info_bg"], C["info_fg"]),
        "progression_free": (C["info_bg"], C["info_fg"]),
    }
    return mapping.get(mode_key, (C["bg2"], C["text2"]))


class RecentAnalysisItem(ctk.CTkFrame):
    def __init__(self, parent, item: dict, on_open=None, **kwargs):
        super().__init__(parent, fg_color="transparent", corner_radius=10, **kwargs)
        self._item = item
        self._on_open = on_open

        result_bg, result_fg = result_style(item.get("result_kind", ""))
        icon_bg, icon_fg = mode_style(item.get("mode_key", ""))

        row = ctk.CTkFrame(
            self, 
            fg_color=C["bg1"],
            corner_radius=10,
            border_width=1,
            border_color=C["bg1"],
            )
        row.pack(fill="x", padx=4, pady=6)
        self._row = row

        icon_box = ctk.CTkFrame(
            row,
            fg_color=icon_bg,
            corner_radius=8,
            width=42,
            height=42
        )
        icon_box.pack(side="left", padx=(0, 14))
        icon_box.pack_propagate(False)

        ctk.CTkLabel(
            icon_box,
            text=mode_icon(item.get("mode_key", "")),
            font=(C["sans"], 16, "bold"),
            text_color=icon_fg,
        ).place(relx=.5, rely=.5, anchor="center")

        txt = ctk.CTkFrame(row, fg_color="transparent")
        txt.pack(side="left", fill="x", expand=True)

        name_lbl = ctk.CTkLabel(
            txt,
            text=item.get("image_name", ""),
            font=(C["sans"], 16, "bold"),
            text_color=C["text1"],
            anchor="w",
            cursor=CLICK_CURSOR,
        )
        name_lbl.pack(fill="x", anchor="w")

        meta_lbl = ctk.CTkLabel(
            txt,
            text=f"{item.get('mode_label', '')}  ·  {format_relative_time(item.get('timestamp', ''))}",
            font=(C["mono"], 10),
            text_color=C["text3"],
            anchor="w",
            cursor=CLICK_CURSOR,
        )
        meta_lbl.pack(fill="x", anchor="w", pady=(2, 0))

        badge = ctk.CTkLabel(
            row,
            text=item.get("result_label", ""),
            font=(C["mono"], 11, "bold"),
            text_color=result_fg,
            fg_color=result_bg,
            corner_radius=18,
            padx=18,
            pady=8,
            cursor=CLICK_CURSOR,
        )
        badge.pack(side="right")

        divider = ctk.CTkFrame(
            self,
            height=1,
            fg_color=C["border"]
        )
        divider.pack(fill="x", padx=4, pady=(6, 0))

        if self._on_open:
            for w in [self, row, txt, name_lbl, meta_lbl, badge, icon_box]:
                w.bind("<Enter>", self._hover_on)
                w.bind("<Leave>", self._hover_off)
                w.bind("<Button-1>", self._handle_open)
                try:
                    w.configure(cursor=CLICK_CURSOR)
                except Exception:
                    pass

    def _handle_open(self, _event=None):
        if self._on_open:
            self._on_open(self._item)
    
    def _hover_on(self, _event=None):
        self._row.configure(
            fg_color=C["bg2"],
            border_color=C["bg1"],
        )

    def _hover_off(self, _event=None):
        self._row.configure(
            fg_color=C["bg1"],
            border_color=C["bg1"],
        )