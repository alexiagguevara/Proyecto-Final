"""Home screen: mode cards + recent analyses list."""
import customtkinter as ctk
from widgets import C, Topbar, ModeCard, RecentAnalysisItem, CLICK_CURSOR
from recent_store import load_recent_analyses, clear_recent_analyses


class HomeView(ctk.CTkFrame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, fg_color=C["bg1"],
                         corner_radius=0, **kwargs)
        self.app = app
        self._build()

    def refresh(self, **_kwargs):
        for w in self._recent_box.winfo_children():
            w.destroy()

        items = load_recent_analyses()

        if not items:
            self._clear_history_btn.pack_forget()
            ctk.CTkLabel(
                self._recent_box,
                text="No recent analyses yet.",
                font=(C["sans"], 12),
                text_color=C["text3"],
            ).pack(anchor="w", pady=(4, 8))
        else:
            if not self._clear_history_btn.winfo_ismapped():
                self._clear_history_btn.pack(side="right")
            for item in items[:5]:
                RecentAnalysisItem(
                    self._recent_box,
                    item,
                    on_open=self._open_recent_analysis
                ).pack(fill="x")

    def _open_recent_analysis(self, item: dict):
        mode_key = item.get("mode_key")

        if mode_key == "classifier":
            self.app.show_view("classifier", recent_item=item)
        elif mode_key in {"progression_anchored", "progression_free"}:
            self.app.show_view("progression", recent_item=item)

    def _clear_recent_history(self):
        clear_recent_analyses()
        self.refresh()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        # Topbar (no back button on home)
        tb = Topbar(self)
        tb.pack(fill="x")

        body = ctk.CTkScrollableFrame(self, fg_color="transparent", scrollbar_button_color="#E5E7EB",
            scrollbar_button_hover_color="#D1D5DB")
        body.pack(fill="both", expand=True)

        # Hero
        hero = ctk.CTkFrame(body, fg_color="transparent")
        hero.pack(fill="x", pady=(32, 20), padx=20)

        ctk.CTkLabel(
            hero,
            text="Astrocyte morphology analysis",
            font=(C["sans"], 20, "bold"),
            text_color=C["text1"],
        ).pack()

        ctk.CTkLabel(
            hero,
            text="Quantify GFAP filament network changes\nfrom epifluorescence microscopy images",
            font=(C["sans"], 13),
            text_color=C["text2"],
            justify="center",
        ).pack(pady=(4, 0))

        # Mode cards grid
        cards = ctk.CTkFrame(body, fg_color="transparent")
        cards.pack(fill="x", padx=16, pady=(0, 28))
        cards.columnconfigure((0, 1), weight=1)

        ModeCard(
            cards,
            title="State Classifier",
            description="Binary classification:\ncontrol vs. pro-inflammatory",
            icon_char="◫",
            icon_bg=C["success_bg"],
            command=lambda: self.app.show_view("classifier"),
        ).grid(row=0, column=0, padx=6, sticky="nsew")

        ModeCard(
            cards,
            title="Progression Profiler",
            description=(
                "Continuous pro-inflammatory score:\n"
                "pro-inflammatory or recovery trajectory"
            ),
            icon_char="↗",
            icon_bg=C["info_bg"],
            command=lambda: self.app.show_view("progression"),
        ).grid(row=0, column=1, padx=6, sticky="nsew")

        # Recent analyses
        recent_header = ctk.CTkFrame(body, fg_color="transparent")
        recent_header.pack(fill="x", padx=20, pady=(10, 8))

        ctk.CTkLabel(
            recent_header,
            text="RECENT ANALYSES",
            font=(C["mono"], 10, "bold"),
            text_color=C["text3"],
        ).pack(side="left")

        self._clear_history_btn = ctk.CTkButton(
            recent_header,
            text="Clear history",
            command=self._clear_recent_history,
            fg_color=C["bg2"],
            text_color=C["danger_fg"],
            hover_color=C["bg3"],
            border_width=1,
            border_color=C["danger_bg"],
            font=(C["sans"], 11, "bold"),
            height=28,
            cursor=CLICK_CURSOR,
        )
        self._clear_history_btn.pack(side="right")

        self._recent_box = ctk.CTkFrame(body, fg_color="transparent")
        self._recent_box.pack(fill="x", padx=20)

        self.refresh()