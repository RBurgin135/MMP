from tkinter.ttk import *


class TopBar(LabelFrame):
    def __init__(self, master, controller, title, *args, **kwargs):
        super().__init__(master, text="top navigation bar", *args, **kwargs)

        # content
        # title
        Label(
            master=self,
            text=title
        ).grid(
            column=1,
            row=0
        )
