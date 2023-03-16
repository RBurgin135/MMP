from tkinter.ttk import *
from GUI.components.util import DirectoryPathButton


class PathFrame(Frame):
    def __init__(self, master, controller, text, text_variable, **kwargs):
        super().__init__(master, **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # content
        Label(
            master=self,
            text=text
        ).grid(
            column=0,
            row=0
        )
        Entry(
            master=self,
            width=50,
            textvariable=text_variable
        ).grid(
            column=1,
            row=0
        )
        DirectoryPathButton(
            master=self,
            text="Find",
            text_variable=text_variable
        ).grid(
            column=2,
            row=0
        )
