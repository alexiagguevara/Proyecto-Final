"""
State Classifier view.
Manages two sub-panels:
  - _input_panel  : upload zone + model-feature reference
  - _result_panel : prediction, probability bars, feature cards, collapsibles
"""
from __future__ import annotations
import threading
import customtkinter as ctk
from numpy import interp
from widgets import (
    C, Topbar, UploadZone, ProbBar, FeatureCard,
    ScoreBar, ProcessingOutputs, FeatureTable, Banner, make_responsive_wrap,
)
from core.pipeline import ClassifierResult, run_classifier
from pathlib import Path
from recent_store import save_recent_analysis, load_asset_image


class ClassifierView(ctk.CTkFrame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, fg_color=C["bg1"],
                         corner_radius=0, **kwargs)
        self.app = app
        self._result: ClassifierResult | None = None
        self._selected_image_path: str | None = None
        self._upload_zone = None
        self._analyze_btn = None
        self._spinner_frame = None
        self._progress = None
        self._selected_file_lbl = None
        self._loading_host = None
        self._build()

    def refresh(self, recent_item=None, **_kwargs):
        if recent_item is not None:
            self._render_recent_result(recent_item)
            self._show("result")
            return
        
        self._reset_input_state()
        self._show("input")

    # ── Build both panels ─────────────────────────────────────────────────────

    def _build(self):
        self._input_panel  = self._build_input()
        self._result_panel = self._build_result()
        self._show("input")

    def _show(self, panel: str):
        for p in (self._input_panel, self._result_panel):
            p.pack_forget()
        target = self._input_panel if panel == "input" else self._result_panel
        target.pack(fill="both", expand=True)

    # ── Input panel ───────────────────────────────────────────────────────────

    def _build_input(self) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(self, fg_color=C["bg1"], corner_radius=0)

        Topbar(panel,
               breadcrumb="State Classifier",
               back_label="Home",
               back_cmd=lambda: self.app.go_back(),
               ).pack(fill="x")

        body = ctk.CTkScrollableFrame(
            panel, 
            fg_color="transparent",
            scrollbar_button_color="#E5E7EB",
            scrollbar_button_hover_color="#D1D5DB",
        )
        body.pack(fill="both", expand=True, padx=20, pady=16)

        desc_lbl = ctk.CTkLabel(
            body,
            text=(
                "Upload a single epifluorescence image.\n"
                "The model classifies astrocyte morphology as control or "
                "pro-inflammatory based on three selectedmorphological features"
                "extracted from the GFAP network."
            ),
            font=(C["sans"], 13),
            text_color=C["text2"],
            justify="left",
            anchor="w",
        )
        desc_lbl.pack(fill="x", anchor="w", pady=(0, 16))
        make_responsive_wrap(desc_lbl, body, padding=40, min_wrap=260)

        self._upload_zone = UploadZone(
            body,
            on_file=self._on_select_file,
            on_clear=self._on_clear_selected_file,
        )
        self._upload_zone.pack(fill="x", pady=(0, 16))


        # Model features reference
        info = ctk.CTkFrame(body, fg_color=C["bg2"],
                            corner_radius=8)
        info.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(info, text="MODEL FEATURES",
                     font=(C["mono"], 10, "bold"),
                     text_color=C["text3"]).pack(
            anchor="w", padx=14, pady=(10, 6))

        for feat, direction in [
            ("median_thickness",      "↓ decreases with inflammation"),
            ("median_tortuosity",     "↑ increases with inflammation"),
            ("median_segment_length", "↑ increases with inflammation"),
        ]:
            row = ctk.CTkFrame(info, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=2)
            ctk.CTkFrame(row, width=7, height=7, corner_radius=4,
                         fg_color=C["green_mid"]).pack(
                side="left", padx=(0, 10), pady=6)
            ctk.CTkLabel(row, text=feat, font=(C["mono"], 12),
                         text_color=C["text1"]).pack(side="left")
            ctk.CTkLabel(row, text=direction, font=(C["sans"], 11),
                         text_color=C["text3"]).pack(side="right")

        ctk.CTkFrame(info, height=10, fg_color="transparent").pack()

        self._analyze_btn = ctk.CTkButton(
            body, text="Analyze image →",
            command=self._on_analyze_click,
            fg_color=C["success_bg"],
            text_color=C["success_fg"],
            hover_color=C["green_light"],
            font=(C["sans"], 14, "bold"),
            height=44,
            state="disabled",
        )
        self._analyze_btn.pack(fill="x", pady=(0, 10))
        self._loading_host = ctk.CTkFrame(body, fg_color="transparent")
        self._loading_host.pack(fill="x")

        return panel

    def _on_select_file(self, path: str):
        self._selected_image_path = path
        if self._analyze_btn is not None:
            self._analyze_btn.configure(state="normal")

    def _on_clear_selected_file(self):
        self._selected_image_path = None
        if self._analyze_btn is not None:
            self._analyze_btn.configure(state="disabled")

    def _on_analyze_click(self):
        if not self._selected_image_path:
            return
        self._run_analysis(self._selected_image_path)

    # ── Result panel ──────────────────────────────────────────────────────────

    def _build_result(self) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(self, fg_color=C["bg1"], corner_radius=0)

        Topbar(panel,
               breadcrumb="State Classifier · Result",
               back_label="Home",
               back_cmd=lambda: self.app.go_back(),
        ).pack(fill="x")

        self._result_body = ctk.CTkScrollableFrame(panel, fg_color="transparent", scrollbar_button_color="#E5E7EB", scrollbar_button_hover_color="#D1D5DB")
        self._result_body.pack(fill="both", expand=True, padx=20, pady=16)

        return panel

    def _render_result(self, r: ClassifierResult):
        """Populate the result panel from a ClassifierResult."""
        for w in self._result_body.winfo_children():
            w.destroy()

        body = self._result_body
        is_ctrl = r.prediction == "Control"
        verdict_color = C["success_fg"] if is_ctrl else C["danger_fg"]

        # Predicted state
        ctk.CTkLabel(body, text="PREDICTED STATE",
                     font=(C["mono"], 10, "bold"),
                     text_color=C["text3"]).pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(body, text=r.prediction,
                     font=(C["sans"], 26, "bold"),
                     text_color=verdict_color).pack(anchor="w")

        interpretation = (
            "The image is morphologically closer to the control condition."
            if is_ctrl
            else "The image is morphologically closer to the inflamed condition."
        )

        interp_lbl = ctk.CTkLabel(
            body,
            text=interpretation,
            font=(C["sans"], 12),
            text_color=C["text2"],
            justify="left",
            anchor="w",
        )
        interp_lbl.pack(fill="x", anchor="w", pady=(4, 12))
        make_responsive_wrap(interp_lbl, body, padding=40, min_wrap=240)

        # Probability bars
        prob_frame = ctk.CTkFrame(body, fg_color="transparent")
        prob_frame.pack(fill="x", pady=(10, 16))
        ProbBar(prob_frame, "Control",
                r.prob_control * 100, C["success_fg"]).pack(fill="x")
        ProbBar(prob_frame, "Pro-inflammatory",
                r.prob_proinflam * 100, C["seg_red"]).pack(fill="x")

        # Feature cards (3-col)
        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.pack(fill="x", pady=(0, 14))
        grid.columnconfigure((0, 1, 2), weight=1)
        for col, (lbl, val) in enumerate([
            ("med. thickness",      f"{r.median_thickness:.2f}"),
            ("med. tortuosity",     f"{r.median_tortuosity:.3f}"),
            ("med. seg. length",    f"{r.median_segment_length:.2f}"),
        ]):
            FeatureCard(grid, lbl, val).grid(row=0, column=col,
                                             padx=4, sticky="nsew")

        # Collapsible: processing outputs
        po = ProcessingOutputs(body,
                               wavelet=r.wavelet_img,
                               mask=r.mask_img,
                               skeleton=r.skeleton_img)
        po.pack(fill="x", pady=(0, 8))

        # Collapsible: all features
        ft = FeatureTable(body, features=r.all_features)
        ft.pack(fill="x", pady=(0, 8))

        # Analyze another image button
        ctk.CTkButton(
            body, text="Analyze another image",
            command=lambda: self._back_to_input(),
            fg_color=C["bg2"],
            text_color=C["text1"],
            hover_color=C["bg3"],
            border_width=1,
            border_color=C["border"],
            font=(C["sans"], 13),
            height=40,
        ).pack(fill="x", pady=(8, 0))

    def _back_to_input(self):
        self._reset_input_state()
        self._show("input")

    # ── Pipeline call (threaded) ──────────────────────────────────────────────

    def _show_loading(self, text="Analyzing...", mode="indeterminate", parent=None):
        self._hide_loading()

        host = parent if parent is not None else self._loading_host

        self._spinner_frame = ctk.CTkFrame(host, fg_color="transparent")
        self._spinner_frame.pack(pady=6)

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

    def _run_analysis(self, path: str):
        # Show loading indicator
        self._show("input")
        self._show_loading("Analyzing image...", mode="indeterminate", parent=self._loading_host)
        self.update_idletasks()

        def _run():
            try:
                result = run_classifier(path)
                self.after(0, lambda: self._on_done(result))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                msg = str(exc)
                self.after(0, lambda m=msg: self._on_error(m))

        self.after(50, lambda: threading.Thread(target=_run, daemon=True).start())

    def _reset_input_state(self):
        self._selected_image_path = None

        if self._upload_zone is not None:
            self._upload_zone.clear()

        if self._analyze_btn is not None:
            self._analyze_btn.configure(state="disabled")

        self._hide_loading()

    def _on_done(self, result: ClassifierResult):
        self._hide_loading()
        self._result = result
        image_name = Path(self._selected_image_path).name if self._selected_image_path else "Unknown"

        result_kind = "Control" if result.prediction == "Control" else "Pro-inflammatory"

        save_recent_analysis(
            image_name=image_name,
            mode_key="classifier",
            mode_label="State Classifier",
            result_label=result.prediction,
            result_kind=result_kind,
            snapshot={
                "prediction": result.prediction,
                "prob_control": result.prob_control,
                "prob_proinflam": result.prob_proinflam,
                "median_thickness": result.median_thickness,
                "median_tortuosity": result.median_tortuosity,
                "median_segment_length": result.median_segment_length,
                "all_features": result.all_features,
                "wavelet_img": result.wavelet_img,
                "mask_img": result.mask_img,
                "skeleton_img": result.skeleton_img,
            }
        )
        self._render_result(result)
        self._show("result")

    def _render_recent_result(self, item: dict):
        snap = item.get("snapshot", {})
        assets = item.get("assets", {})

        result = ClassifierResult(
            prediction=snap.get("prediction", ""),
            prob_control=float(snap.get("prob_control", 0.0)),
            prob_proinflam=float(snap.get("prob_proinflam", 0.0)),
            median_thickness=float(snap.get("median_thickness", 0.0)),
            median_tortuosity=float(snap.get("median_tortuosity", 0.0)),
            median_segment_length=float(snap.get("median_segment_length", 0.0)),
            all_features=snap.get("all_features", {}),
            wavelet_img=load_asset_image(assets.get("wavelet")),
            mask_img=load_asset_image(assets.get("mask")),
            skeleton_img=load_asset_image(assets.get("skeleton")),
        )

        self._result = result
        self._render_result(result)

    def _on_error(self, msg: str):
        self._hide_loading()
        self._clear_input_messages()
        Banner(self._input_panel,
               text=f"Error: {msg}",
               style="danger").pack(padx=20, pady=8)

    def _clear_input_messages(self):
        for child in self._input_panel.winfo_children():
            if isinstance(child, Banner):
                child.destroy()