import tkinter as tk
from ctypes import windll
from GUI.screens.model_screen import model_screen

windll.shcore.SetProcessDpiAwareness(1)

root = tk.Tk()
root.title("PCA Wavelet Model Manager")
root.iconbitmap('GUI/assets/kingfisher.ico')

model_screen(root)

if __name__ == "__main__":
    root.mainloop()
