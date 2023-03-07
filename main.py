from tkinter import *
from ctypes import windll

from GUI.screens.model_screen import ModelScreen
from GUI.screens.training_screen import TrainingScreen

windll.shcore.SetProcessDpiAwareness(1)


class MMP(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.title("PCA Wavelet Model Manager")
        self.iconbitmap('GUI/assets/kingfisher.ico')
        self.geometry("1000x600")

        # initialise root frame
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # initialise frames to empty set
        self.frames = {}
        for Screen in (ModelScreen, TrainingScreen):
            frame = Screen(self, self)
            self.frames[Screen] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # show first screen
        self.show_frame(ModelScreen)

    def show_frame(self, key):
        frame = self.frames[key]
        frame.tkraise()


if __name__ == "__main__":
    app = MMP()
    app.mainloop()
