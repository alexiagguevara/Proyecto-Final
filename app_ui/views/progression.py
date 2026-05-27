"""
Progression Profiler view.
Sub-panels:
  _anchored_input   — tab "Experiment-Anchored": ref upload + analyze upload
  _free_input       — tab "Reference-Free":  single analyze upload
  _anchored_result  — recovery score (red → green), collapsibles
  _free_result      — inflammatory score (green → red) + thresholds, collapsibles
"""
from __future__ import annotations
import threading
import customtkinter as ctk
from widgets import (
    C, Topbar, UploadZone, RefUploadZone, ScoreBar,
    FeatureCard, ProcessingOutputs, FeatureTable, Banner, WrappedBanner, CollapsibleSection, make_responsive_wrap,
)
from core.pipeline import (
    ProgressionResult,
    compute_anchors,
    run_progression_anchored,
    run_progression_absolute,
)
from pathlib import Path
from recent_store import save_recent_analysis, load_asset_image

# Recovery category labels (inverted thresholds)
RECOVERY_CATEGORIES = [
    (-10_000,     18.25, "minimal recovery"),
    (18.25, 51.44, "partial recovery"),
    (51.44, 83.19, "substantial recovery"),
    (83.19, 10_000, "near-complete recovery"),
]

INFLAM_CATEGORIES = [
    (-10_000, 16.81, "low inflammatory morphology"),
    (16.81,   48.56, "mild inflammatory morphology"),
    (48.56,   81.75, "moderate inflammatory morphology"),
    (81.75,   10_000, "high inflammatory morphology"),
]


def _category(score: float, table: list) -> str:
    for lo, hi, label in table:
        if lo <= score < hi:
            return label
    return table[-1][2]

def _recovery_color(score: float) -> str:
    if score < 18.25:
        return "#E24B4A"   # red
    elif score < 51.44:
        return "#D85A30"   # coral
    elif score < 83.19:
        return "#BA7517"   # amber
    else:
        return "#3B6D11"   # green


def _inflammatory_color(score: float) -> str:
    if score < 16.81:
        return "#3B6D11"   # green
    elif score < 48.56:
        return "#BA7517"   # amber
    elif score < 81.75:
        return "#D85A30"   # coral
    else:
        return "#E24B4A"   # red


class ProgressionView(ctk.CTkFrame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, fg_color=C["bg1"],
                         corner_radius=0, **kwargs)
        self.app = app
        self._ctrl_paths:   list[str] = []
        self._proinf_paths: list[str] = []
        self._ctrl_anchor   = None
        self._proinf_anchor = None
        self._result: ProgressionResult | None = None
        self._anchored_target_path: str | None = None
        self._free_target_path: str | None = None

        self._anchored_upload_zone = None
        self._free_upload_zone = None
        
        self._anchored_loading_host = None
        self._free_loading_host = None
        
        self._spinner_frame = None
        self._progress = None

        self._anchored_analyze_btn = None
        self._free_analyze_btn = None

        self._anchored_file_lbl = None
        self._free_file_lbl = None
        self._build()

    def _reset_anchored_state(self):
        self._ctrl_paths = []
        self._proinf_paths = []
        self._anchored_target_path = None

        if hasattr(self, "_ctrl_upload") and self._ctrl_upload is not None:
            self._ctrl_upload.reset()

        if hasattr(self, "_proinf_upload") and self._proinf_upload is not None:
            self._proinf_upload.reset()

        if self._anchored_upload_zone is not None:
            self._anchored_upload_zone.clear()

        if self._anchored_analyze_btn is not None:
            self._anchored_analyze_btn.configure(state="disabled")

        self._hide_loading()

    def _reset_free_state(self):
        self._free_target_path = None

        if self._free_upload_zone is not None:
            self._free_upload_zone.clear()

        if self._free_analyze_btn is not None:
            self._free_analyze_btn.configure(state="disabled")

        self._hide_loading()
    
    def refresh(self, recent_item=None, **_kwargs):
        if recent_item is not None:
            self._render_recent_result(recent_item)
            return

        self._reset_anchored_state()
        self._reset_free_state()
        self._show("anchored_input")

    # ── Panel management ──────────────────────────────────────────────────────

    def _show(self, panel: str):
        all_panels = [
            self._anchored_input,
            self._free_input,
            self._anchored_result,
            self._free_result,
        ]
        for p in all_panels:
            p.pack_forget()
        target = {
            "anchored_input":   self._anchored_input,
            "free_input":       self._free_input,
            "anchored_result":  self._anchored_result,
            "free_result":      self._free_result,
        }[panel]
        target.pack(fill="both", expand=True)
        self._scroll_to_top(target)

    # ── Build all panels ──────────────────────────────────────────────────────

    def _build(self):
        self._anchored_input  = self._build_anchored_input()
        self._free_input      = self._build_free_input()
        self._anchored_result = self._build_result_panel("anchored")
        self._free_result     = self._build_result_panel("free")
        self._show("anchored_input")

    # ── Shared topbar helper ──────────────────────────────────────────────────

    def _topbar(self, panel, breadcrumb: str, back_target: str):
        Topbar(panel,
               breadcrumb=breadcrumb,
               back_label="Home" if back_target == "Home" else "New image",
               back_cmd=(lambda: self.app.go_back()
                         if back_target == "Home"
                         else lambda: self._show(back_target)),
               ).pack(fill="x")

    # ── Anchored input ────────────────────────────────────────────────────────

    def _build_anchored_input(self) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(self, fg_color=C["bg1"], corner_radius=0)

        Topbar(panel,
               breadcrumb="Progression Profiler",
               back_label="Home",
               back_cmd=lambda: self.app.go_back(),
               ).pack(fill="x")

        # Tab bar
        tab_bar = ctk.CTkFrame(panel, fg_color="transparent")
        tab_bar.pack(fill="x", padx=16, pady=(14, 0))
        tab_bar.columnconfigure((0, 1), weight=1)

        self._tab_anchored_btn = ctk.CTkButton(
            tab_bar, text="Experiment-Anchored\nrecommended",
            command=lambda: self._show("anchored_input"),
            fg_color=C["info_bg"],
            text_color=C["info_fg"],
            hover_color=C["info_bg"],
            font=(C["sans"], 12, "bold"),
            height=46,
        )
        self._tab_anchored_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(
            tab_bar, text="Reference-Free\nfallback",
            command=lambda: self._show("free_input"),
            fg_color=C["bg2"],
            text_color=C["text2"],
            hover_color=C["bg3"],
            border_width=1, border_color=C["border"],
            font=(C["sans"], 12),
            height=46,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        body = ctk.CTkScrollableFrame(
            panel, fg_color="transparent",
            scrollbar_button_color="#E5E7EB",
            scrollbar_button_hover_color="#D1D5DB",
        )
        body.pack(fill="both", expand=True, padx=(20, 8), pady=14)

        WrappedBanner(body,
               text=(
                   "Provide control and inflamed reference images from the same experiment. "
                   "This mode computes a recovery-oriented score anchored to your own references, reducing between-experiment variability."
               ),
               style="info").pack(fill="x", pady=(0, 14))

        info = ctk.CTkFrame(body, fg_color=C["bg1"], corner_radius=8)
        info.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(
            info,
            text="SCORE FEATURES",
            font=(C["mono"], 12, "bold"),
            text_color=C["text3"],
        ).pack(anchor="w", padx=14, pady=(10, 6))

        for feat, direction in [
            ("median_thickness", "↑ increases with recovery"),
            ("median_segment_length", "↓ decreases with recovery"),
        ]:
            row = ctk.CTkFrame(info, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=2)

            ctk.CTkFrame(
                row,
                width=7,
                height=7,
                corner_radius=4,
                fg_color=C["green_mid"],
            ).pack(side="left", padx=(0, 10), pady=6)

            ctk.CTkLabel(
                row,
                text=feat,
                font=(C["mono"], 12),
                text_color=C["text1"],
            ).pack(side="left")

            ctk.CTkLabel(
                row,
                text=direction,
                font=(C["sans"], 11),
                text_color=C["text3"],
            ).pack(side="right")

        ctk.CTkFrame(info, height=10, fg_color="transparent").pack()

        ctk.CTkLabel(body,
                     text="REFERENCE IMAGES — add as many as you have",
                     font=(C["mono"], 12, "bold"),
                     text_color=C["text3"]).pack(anchor="w", pady=(0, 8))

        # Control refs
        self._ctrl_upload = RefUploadZone(
            body,
            label="Control images (0 h)",
            on_files=self._on_ctrl_files,
            on_change=self._on_ctrl_refs_changed,
        )
        self._ctrl_upload.pack(fill="x", pady=(0, 8))

        # Pro-inflammatory refs
        self._proinf_upload = RefUploadZone(
            body,
            label="Pro-inflammatory images (e.g. 72 h LPS)",
            on_files=self._on_proinf_files,
            on_change=self._on_proinf_refs_changed,
        )
        self._proinf_upload.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(body,
                     text="IMAGE TO ANALYZE",
                     font=(C["mono"], 12, "bold"),
                     text_color=C["text3"]).pack(anchor="w", pady=(0, 8))

        self._anchored_upload_zone = UploadZone(
            body,
            on_file=self._on_select_anchored_target,
            on_clear=self._on_clear_anchored_target,
        )
        self._anchored_upload_zone.pack(fill="x", pady=(0, 12))

        self._anchored_analyze_btn = ctk.CTkButton(
            body,
            text="Analyze image →",
            command=self._on_anchored_analyze_click,
            fg_color=C["success_bg"],
            text_color=C["success_fg"],
            hover_color=C["green_light"],
            font=(C["sans"], 14, "bold"),
            height=44,
            state="disabled",
        )
        self._anchored_analyze_btn.pack(fill="x", pady=(0, 8))

        self._anchored_loading_host = ctk.CTkFrame(body, fg_color="transparent")
        self._anchored_loading_host.pack(fill="x")

        return panel
    
    def _on_select_anchored_target(self, path: str):
        self._anchored_target_path = path
        self._update_anchored_analyze_state()

    def _on_clear_anchored_target(self):
        self._anchored_target_path = None
        self._update_anchored_analyze_state()

    def _on_anchored_analyze_click(self):
        if not self._anchored_target_path:
            return
        self._run_anchored_analysis(self._anchored_target_path)

    # ── Free input ────────────────────────────────────────────────────────────

    def _build_free_input(self) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(self, fg_color=C["bg1"], corner_radius=0)

        Topbar(panel,
               breadcrumb="Progression Profiler",
               back_label="Home",
               back_cmd=lambda: self.app.go_back(),
               ).pack(fill="x")

        # Tab bar
        tab_bar = ctk.CTkFrame(panel, fg_color="transparent")
        tab_bar.pack(fill="x", padx=16, pady=(14, 0))
        tab_bar.columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            tab_bar, text="Experiment-Anchored\nrecommended",
            command=lambda: self._show("anchored_input"),
            fg_color=C["bg2"],
            text_color=C["text2"],
            hover_color=C["bg3"],
            border_width=1, border_color=C["border"],
            font=(C["sans"], 12),
            height=46,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(
            tab_bar, text="Reference-Free\nfallback",
            command=lambda: self._show("free_input"),
            fg_color=C["info_bg"],
            text_color=C["info_fg"],
            hover_color=C["info_bg"],
            font=(C["sans"], 12, "bold"),
            height=46,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        body = ctk.CTkScrollableFrame(
            panel, fg_color="transparent",
            scrollbar_button_color="#E5E7EB",
            scrollbar_button_hover_color="#D1D5DB",
        )
        body.pack(fill="both", expand=True, padx=(20, 8), pady=14)

        WrappedBanner(body,
               text=(
                   "No experiment references needed. Score is calculated against "
                   "population-level anchors from the training dataset. "
                   "Results may vary across experimental setups."
               ),
               style="warning").pack(fill="x", pady=(0, 14))

        info = ctk.CTkFrame(body, fg_color=C["bg1"], corner_radius=8)
        info.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(
            info,
            text="SCORE FEATURES",
            font=(C["mono"], 12, "bold"),
            text_color=C["text3"],
        ).pack(anchor="w", padx=14, pady=(10, 6))

        for feat, direction in [
            ("median_thickness", "↓ decreases with inflammatory progression"),
            ("median_segment_length", "↑ increases with inflammatory progression"),
        ]:
            row = ctk.CTkFrame(info, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=2)

            ctk.CTkFrame(
                row,
                width=7,
                height=7,
                corner_radius=4,
                fg_color=C["green_mid"],
            ).pack(side="left", padx=(0, 10), pady=6)

            ctk.CTkLabel(
                row,
                text=feat,
                font=(C["mono"], 12),
                text_color=C["text1"],
            ).pack(side="left")

            ctk.CTkLabel(
                row,
                text=direction,
                font=(C["sans"], 11),
                text_color=C["text3"],
            ).pack(side="right")

        ctk.CTkFrame(info, height=10, fg_color="transparent").pack()

        self._free_upload_zone = UploadZone(
            body,
            label="Select .tif image",
            icon_color=C["success_fg"],
            on_file=self._on_select_free_target,
            on_clear=self._on_clear_free_target,
        )
        self._free_upload_zone.pack(fill="x", pady=(0, 12))

        self._free_analyze_btn = ctk.CTkButton(
            body,
            text="Analyze image →",
            command=self._on_free_analyze_click,
            fg_color=C["success_bg"],
            text_color=C["success_fg"],
            hover_color=C["green_light"],
            font=(C["sans"], 14, "bold"),
            height=44,
            state="disabled",
        )
        self._free_analyze_btn.pack(fill="x", pady=(0, 8))

        self._free_loading_host = ctk.CTkFrame(body, fg_color="transparent")
        self._free_loading_host.pack(fill="x")

        return panel
    
    def _on_select_free_target(self, path: str):
        self._free_target_path = path
        if self._free_analyze_btn is not None:
            self._free_analyze_btn.configure(state="normal")

    def _on_clear_free_target(self):
        self._free_target_path = None
        if self._free_analyze_btn is not None:
            self._free_analyze_btn.configure(state="disabled")

    def _on_free_analyze_click(self):
        if not self._free_target_path:
            return
        self._run_free_analysis(self._free_target_path)

    # ── Result panel (shared template for both modes) ─────────────────────────

    def _build_result_panel(self, mode: str) -> ctk.CTkFrame:
        """Build a blank result panel; content is populated at render time."""
        panel = ctk.CTkFrame(self, fg_color=C["bg1"], corner_radius=0)

        bc = ("Progression Profiler · Anchored · Result"
              if mode == "anchored"
              else "Progression Profiler · Reference-free · Result")
        if mode == "anchored":
            back_cmd = self._back_to_anchored_input
        else:
            back_cmd = self._back_to_free_input

        Topbar(panel,
               breadcrumb=bc,
               back_label="Home",
               back_cmd=lambda: self.app.go_back(),
               ).pack(fill="x")

        # Scrollable body placeholder — filled in _render_result()
        body_holder = ctk.CTkScrollableFrame(
            panel, fg_color="transparent",
            scrollbar_button_color="#E5E7EB",
            scrollbar_button_hover_color="#D1D5DB",
        )
        body_holder.pack(fill="both", expand=True, padx=(20, 8), pady=16)

        if mode == "anchored":
            self._anchored_result_body = body_holder
        else:
            self._free_result_body = body_holder

        return panel

    # ── Render helpers ────────────────────────────────────────────────────────

    def _render_anchored_result(self, r: ProgressionResult):
        body = self._anchored_result_body
        for w in body.winfo_children():
            w.destroy()

        inner = ctk.CTkFrame(body, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=0, pady=0)

        recovery = r.recovery_score
        category = _category(recovery, RECOVERY_CATEGORIES)

        ctk.CTkLabel(inner, text="RECOVERY SCORE",
                     font=(C["mono"], 12, "bold"),
                     text_color=C["text3"]).pack(anchor="w", pady=(0, 8))

        bar = ScoreBar(inner, score=recovery, mode="recovery", bg=C["bg1"])
        bar.pack(fill="x", pady=(0, 4))

        score_color = _recovery_color(recovery)

        ctk.CTkLabel(inner, text=f"{recovery:.1f}",
                     font=(C["sans"], 34, "bold"),
                     text_color=score_color).pack(anchor="w")
        ctk.CTkLabel(inner, text=category,
                     font=(C["sans"], 13),
                     text_color=C["text2"]).pack(anchor="w", pady=(0, 4))
        interp_lbl = ctk.CTkLabel(
            inner,
            text="Interpretation: higher recovery indicates morphology closer to the control references.",
            font=(C["sans"], 12),
            text_color=C["text2"],
            justify="left",
            anchor="w",
        )
        interp_lbl.pack(fill="x", anchor="w", pady=(0, 10))
        make_responsive_wrap(interp_lbl, inner, padding=40, min_wrap=240)

        # Anchor note
        if r.note:
            note_lbl = ctk.CTkLabel(
                inner,
                text=r.note,
                font=(C["mono"], 11),
                text_color=C["text3"],
                justify="left",
                anchor="w",
            )
            note_lbl.pack(fill="x", anchor="w", pady=(0, 14))
            make_responsive_wrap(note_lbl, inner, padding=40, min_wrap=240)
        else:
            ctk.CTkLabel(
                inner,
                text=(f"anchored to {r.n_ctrl_refs} ctrl  and  "
                    f"{r.n_proinf_refs} pro-inflammatory images"),
                font=(C["mono"], 10),
                text_color=C["text3"],
            ).pack(anchor="w", pady=(0, 14))

        # Dual cards
        cards = ctk.CTkFrame(inner, fg_color="transparent")
        cards.pack(fill="x", pady=(0, 14))
        cards.columnconfigure((0, 1), weight=1)
        FeatureCard(cards, "Recovery score",      f"{r.recovery_score:.1f}").grid(
            row=0, column=0, padx=(0, 4), sticky="nsew")
        FeatureCard(cards, "Inflammatory score",  f"{r.inflammatory_score:.1f}").grid(
            row=0, column=1, padx=(4, 0), sticky="nsew")

        feat_cards = ctk.CTkFrame(inner, fg_color="transparent")
        feat_cards.pack(fill="x", pady=(0, 14))
        feat_cards.columnconfigure((0, 1), weight=1)

        FeatureCard(feat_cards, "Med. thickness", f"{r.median_thickness:.2f}").grid(
            row=0, column=0, padx=(0, 4), sticky="nsew"
        )
        FeatureCard(feat_cards, "Med. seg. length", f"{r.median_segment_length:.2f}").grid(
            row=0, column=1, padx=(4, 0), sticky="nsew"
        )
        
        ProcessingOutputs(inner,
                          wavelet=r.wavelet_img,
                          mask=r.mask_img,
                          skeleton=r.skeleton_img).pack(fill="x", pady=(0, 8))

        FeatureTable(inner, features=r.all_features).pack(fill="x", pady=(0, 8))

        ctk.CTkButton(
            inner, text="Analyze another image",
            command=self._back_to_anchored_input,
            fg_color=C["bg2"],
            text_color=C["text1"],
            hover_color=C["bg3"],
            border_width=1, border_color=C["border"],
            font=(C["sans"], 13),
            height=40,
        ).pack(fill="x", pady=(4, 0))

    def _render_free_result(self, r: ProgressionResult):
        body = self._free_result_body
        for w in body.winfo_children():
            w.destroy()

        inner = ctk.CTkFrame(body, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=0, pady=0)

        banner_text = r.warning if r.warning else (
            "Population-level anchors used (training dataset). "
            "Consider using Experiment-Anchored mode when references are available."
        )

        WrappedBanner(inner,
               text=banner_text,
               style="warning").pack(fill="x", pady=(0, 14))

        inflam   = r.inflammatory_score
        category = _category(inflam, INFLAM_CATEGORIES)

        ctk.CTkLabel(inner, text="INFLAMMATORY SCORE",
                     font=(C["mono"], 12, "bold"),
                     text_color=C["text3"]).pack(anchor="w", pady=(0, 8))

        bar = ScoreBar(inner, score=inflam, mode="inflammatory", bg=C["bg1"])
        bar.pack(fill="x", pady=(0, 4))

        score_color = _inflammatory_color(inflam)

        ctk.CTkLabel(inner, text=f"{inflam:.1f}",
                     font=(C["sans"], 34, "bold"),
                     text_color=score_color).pack(anchor="w")
        ctk.CTkLabel(inner, text=category,
                     font=(C["sans"], 13),
                     text_color=C["text2"]).pack(anchor="w", pady=(0, 4))
        interp_lbl = ctk.CTkLabel(
            inner,
            text="Interpretation: higher values indicate morphology closer to the inflamed population anchor.",
            font=(C["sans"], 12),
            text_color=C["text2"],
            justify="left",
            anchor="w",
        )
        interp_lbl.pack(fill="x", anchor="w", pady=(0, 14))
        make_responsive_wrap(interp_lbl, inner, padding=40, min_wrap=240)

        # Reference values
        cards = ctk.CTkFrame(inner, fg_color="transparent")
        cards.pack(fill="x", pady=(0, 14))
        cards.columnconfigure((0, 1), weight=1)
        FeatureCard(cards, "Med. thickness",
                    f"{r.median_thickness:.2f}").grid(
            row=0, column=0, padx=(0, 4), sticky="nsew")
        FeatureCard(cards, "Med. seg. length",
                    f"{r.median_segment_length:.2f}").grid(
            row=0, column=1, padx=(4, 0), sticky="nsew")

        ProcessingOutputs(inner,
                          wavelet=r.wavelet_img,
                          mask=r.mask_img,
                          skeleton=r.skeleton_img).pack(fill="x", pady=(0, 8))

        FeatureTable(inner, features=r.all_features).pack(fill="x", pady=(0, 8))

        ctk.CTkButton(
            inner, text="Analyze another image",
            command=self._back_to_free_input,
            fg_color=C["bg2"],
            text_color=C["text1"],
            hover_color=C["bg3"],
            border_width=1, border_color=C["border"],
            font=(C["sans"], 13),
            height=40,
        ).pack(fill="x", pady=(4, 0))

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_ctrl_files(self, paths: list[str]):
        self._ctrl_paths = self._ctrl_upload.get_paths()
        self._update_anchored_analyze_state()

    def _on_proinf_files(self, paths: list[str]):
        self._proinf_paths.extend(paths)
        self._update_anchored_analyze_state()

    def _on_ctrl_refs_changed(self, paths: list[str]):
        self._ctrl_paths = list(paths)
        self._update_anchored_analyze_state()


    def _on_proinf_refs_changed(self, paths: list[str]):
        self._proinf_paths = list(paths)
        self._update_anchored_analyze_state()

    def _update_anchored_analyze_state(self):
        ready = bool(self._ctrl_paths and self._proinf_paths and self._anchored_target_path)
        if self._anchored_analyze_btn is not None:
            self._anchored_analyze_btn.configure(state="normal" if ready else "disabled")
    
    def _scroll_to_top(self, panel_widget):
        try:
            panel_widget._parent_canvas.yview_moveto(0)
        except Exception:
            pass

    def _show_loading(self, text="Analyzing...", mode="indeterminate", parent=None):
        self._hide_loading()

        host = parent if parent is not None else self

        self._spinner_frame = ctk.CTkFrame(host, fg_color="transparent")
        self._spinner_frame.pack(pady=10)

        self._spinner_label = ctk.CTkLabel(
            self._spinner_frame,
            text=text,
            font=(C["sans"], 13),
            text_color=C["text2"],
        )
        self._spinner_label.pack(pady=(0, 6))

        self._progress = ctk.CTkProgressBar(self._spinner_frame, width=220)
        self._progress.pack()
        if mode == "indeterminate":
            self._progress.configure(mode="indeterminate")
            self._progress.start()
        else:
            self._progress.configure(mode="determinate")
            self._progress.set(0.0)
    
    def _hide_loading(self):
        if hasattr(self, "_progress") and self._progress is not None:
            try:
                self._progress.stop()
            except Exception:
                pass
        if hasattr(self, "_spinner_frame") and self._spinner_frame is not None:
            try:
                self._spinner_frame.destroy()
            except Exception:
                pass
            self._spinner_frame = None
            self._progress = None

    def _run_anchored_analysis(self, path: str):
        if not self._ctrl_paths or not self._proinf_paths:
            Banner(self._anchored_input,
                   text="Please load at least one control and one pro-inflammatory reference image first.",
                   style="warning").pack(padx=20, pady=8)
            return

        self._show("anchored_input")
        self._show_loading("Analyzing anchored progression...", mode="indeterminate", parent=self._anchored_loading_host)
        self.update_idletasks()

        def _run():
            try:
                ctrl_anchor, proinf_anchor = compute_anchors(
                    self._ctrl_paths, self._proinf_paths
                )
                r = run_progression_anchored(path, ctrl_anchor, proinf_anchor)
                r.n_ctrl_refs   = len(self._ctrl_paths)
                r.n_proinf_refs = len(self._proinf_paths)
                self.after(0, lambda: self._on_anchored_done(r))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                msg = str(exc)
                self.after(0, lambda m=msg: self._on_error(m, "anchored"))

        threading.Thread(target=_run, daemon=True).start()

    def _run_free_analysis(self, path: str):
        self._show("free_input")
        self._show_loading("Analyzing reference-free progression...", mode="indeterminate", parent=self._free_loading_host)
        self.update_idletasks()
        
        def _run():
            try:
                r = run_progression_absolute(path)
                self.after(0, lambda: self._on_free_done(r))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                msg = str(exc)
                self.after(0, lambda m=msg: self._on_error(m, "free"))

        threading.Thread(target=_run, daemon=True).start()

    def _on_anchored_done(self, r: ProgressionResult):
        self._hide_loading()
        
        image_name = Path(self._anchored_target_path).name if self._anchored_target_path else "Unknown"

        recovery = r.recovery_score
        if recovery < 18.25:
            result_kind = "minimal_recovery"
            result_label = "Minimal recovery"
        elif recovery < 51.44:
            result_kind = "partial_recovery"
            result_label = "Partial recovery"
        elif recovery < 83.19:
            result_kind = "substantial_recovery"
            result_label = "Substantial recovery"
        else:
            result_kind = "near_complete_recovery"
            result_label = "Near-complete recovery"

        save_recent_analysis(
            image_name=image_name,
            mode_key="progression_anchored",
            mode_label="Progression Profiler · Anchored",
            result_label=result_label,
            result_kind=result_kind,
            snapshot={
                "mode": "anchored",
                "inflammatory_score": r.inflammatory_score,
                "recovery_score": r.recovery_score,
                "median_thickness": r.median_thickness,
                "median_segment_length": r.median_segment_length,
                "all_features": r.all_features,
                "note": r.note,
                "warning": r.warning,
                "wavelet_img": r.wavelet_img,
                "mask_img": r.mask_img,
                "skeleton_img": r.skeleton_img,
            }
        )

        self._render_anchored_result(r)
        self._show("anchored_result")

    def _on_free_done(self, r: ProgressionResult):
        self._hide_loading()

        image_name = Path(self._free_target_path).name if self._free_target_path else "Unknown"

        inflam = r.inflammatory_score
        if inflam < 16.81:
            result_kind = "low"
            result_label = "Low inflammatory morph"
        elif inflam < 48.56:
            result_kind = "mild"
            result_label = "Mild inflammatory morph"
        elif inflam < 81.75:
            result_kind = "moderate"
            result_label = "Moderate inflammatory morph"
        else:
            result_kind = "high"
            result_label = "High inflammatory morph"

        save_recent_analysis(
            image_name=image_name,
            mode_key="progression_free",
            mode_label="Progression Profiler · Reference-free",
            result_label=result_label,
            result_kind=result_kind,
            snapshot={
                "mode": "free",
                "inflammatory_score": r.inflammatory_score,
                "recovery_score": r.recovery_score,
                "median_thickness": r.median_thickness,
                "median_segment_length": r.median_segment_length,
                "all_features": r.all_features,
                "note": r.note,
                "warning": r.warning,
                "wavelet_img": r.wavelet_img,
                "mask_img": r.mask_img,
                "skeleton_img": r.skeleton_img,
            }
        )

        self._render_free_result(r)
        self._show("free_result")

    def _render_recent_result(self, item: dict):
        snap = item.get("snapshot", {})
        assets = item.get("assets", {})

        r = ProgressionResult(
            inflammatory_score=float(snap.get("inflammatory_score", 0.0)),
            recovery_score=float(snap.get("recovery_score", 0.0)),
            median_thickness=float(snap.get("median_thickness", 0.0)),
            median_segment_length=float(snap.get("median_segment_length", 0.0)),
            all_features=snap.get("all_features", {}),
            wavelet_img=load_asset_image(assets.get("wavelet")),
            mask_img=load_asset_image(assets.get("mask")),
            skeleton_img=load_asset_image(assets.get("skeleton")),
            note=snap.get("note"),
            warning=snap.get("warning"),
        )

        if snap.get("mode") == "anchored":
            self._hide_loading()
            self._render_anchored_result(r)
            self._show("anchored_result")
        else:
            self._hide_loading()
            self._render_free_result(r)
            self._show("free_result")

    def _on_error(self, msg: str, mode: str):
        self._hide_loading()
        target = (self._anchored_input if mode == "anchored"
                  else self._free_input)
        Banner(target, text=f"Error: {msg}", style="danger").pack(
            padx=20, pady=8)

    def _back_to_anchored_input(self):
        self._reset_anchored_state()
        self._show("anchored_input")

    def _back_to_free_input(self):
        self._reset_free_state()
        self._show("free_input")
    