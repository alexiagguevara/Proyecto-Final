"""Main application window and navigation controller."""
import customtkinter as ctk


class AstroMetrixApp(ctk.CTk):
    WIDTH  = 540
    HEIGHT = 820

    def __init__(self):
        super().__init__()
        self.title("AstroMetrix  v1.0")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.minsize(740, 660)
        self.resizable(True, True)

        # ── Scrollable container that hosts all views stacked at (0,0)
        self._container = ctk.CTkFrame(self, fg_color="white", corner_radius=0)
        self._container.pack(fill="both", expand=True)
        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(0, weight=1)

        # ── Views are imported here to avoid circular imports at module level
        from views.home        import HomeView
        from views.classifier  import ClassifierView
        from views.progression import ProgressionView

        self._views: dict[str, ctk.CTkFrame] = {}
        self._history: list[str] = []
        self._current: str | None = None

        for name, cls in [
            ("home",        HomeView),
            ("classifier",  ClassifierView),
            ("progression", ProgressionView),
        ]:
            v = cls(self._container, app=self)
            v.grid(row=0, column=0, sticky="nsew")
            self._views[name] = v

        self.show_view("home")

    # ── Navigation API ────────────────────────────────────────────────────────

    def show_view(self, name: str, **kwargs) -> None:
        """Bring a view to the front and call its refresh() hook."""
        if self._current and self._current != name:
            self._history.append(self._current)
        self._current = name
        view = self._views[name]
        view.refresh(**kwargs)
        view.tkraise()

    def go_back(self) -> None:
        if self._history:
            self.show_view(self._history.pop())