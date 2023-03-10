from tkinter.ttk import *
from GUI.components.util import FilePathButton


class PathFrame(Frame):
    def __init__(self, master, controller, text, text_variable, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
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
        FilePathButton(
            master=self,
            text="Find",
            filetypes=(("python files", "*.py"), ("all files", "*.*"))
        ).grid(
            column=2,
            row=0
        )
