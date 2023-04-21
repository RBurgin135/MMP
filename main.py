from tkinter import *
from ctypes import windll

from GUI.utility import navigation
from GUI.utility.styling import styling
from Model.model import Model

windll.shcore.SetProcessDpiAwareness(1)


class MMP(Tk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title("PCA Wavelet Model Manager")
        self.iconbitmap('GUI/assets/kingfisher.ico')
        self.geometry("1000x600")
        self.minsize(width=950, height=400)
        self.maxsize(width=1200, height=800)
        self.resizable(width=False, height=False)

        # styling
        styling(self)

        # initialise root frame
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # instantiate data class
        self.current_model = Model(self)

        # show first screen
        self.current_frame = Frame()
        self.navigate("model")

    def navigate(self, route):
        self.current_frame.grid_forget()
        self.current_frame = navigation.Screens[route](
            master=self,
            controller=self,
            current_model=self.current_model
        )
        self.current_frame.grid(sticky="news")


if __name__ == "__main__":
    app = MMP()
    app.mainloop()
