import tkinter as tk
from ctypes import windll
from GUI.screens.model_screen import model_screen

windll.shcore.SetProcessDpiAwareness(1)

root = tk.Tk()
root.title("PCA Wavelet Model Manager")
root.iconbitmap('GUI/assets/kingfisher.ico')
root.geometry("1000x600")

model_screen(root)

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=5)

if __name__ == "__main__":
    root.mainloop()
