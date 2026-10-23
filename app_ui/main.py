"""AstroMetrix — entry point."""
import os
from paths import user_data_dir

# Persistent matplotlib cache: avoids rebuilding the font cache on every launch
os.environ["MPLCONFIGDIR"] = str(user_data_dir() / "mplconfig")
os.environ["MPLBACKEND"] = "Agg"

import customtkinter as ctk
from app import AstroMetrixApp

if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")
    app = AstroMetrixApp()
    app.mainloop()