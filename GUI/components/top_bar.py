from tkinter.ttk import *


class TopBar(LabelFrame):
    def __init__(self, master, controller, title, **kwargs):
        super().__init__(master, text="top navigation bar", **kwargs)

        # content
        # title
        Label(
            master=self,
            text=title
        ).grid(
            column=1,
            row=0
        )
