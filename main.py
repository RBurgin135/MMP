from tkinter import *
from ctypes import windll

from GUI.screens import navigation
from Model.model import Model

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

        # instantiate data class
        current_model = Model(self)

        # initialise frames to empty set
        self.frames = {}
        for Screen in navigation.Screens.values():
            frame = Screen(
                master=self,
                controller=self,
                current_model=current_model
            )
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[Screen] = frame

        # show first screen
        self.navigate("model")

    def navigate(self, route):
        frame = navigation.Screens[route]
        self.frames[frame].tkraise()


if __name__ == "__main__":
    app = MMP()
    app.mainloop()
