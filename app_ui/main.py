"""AstroMetrix — entry point."""
import customtkinter as ctk
from app import AstroMetrixApp

if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")
    app = AstroMetrixApp()
    app.mainloop()