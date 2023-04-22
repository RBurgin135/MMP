import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import *


class DirectoryPathButton(Button):
    def __init__(self, text, text_variable, **kwargs):
        super().__init__(
            text=text,
            command=lambda:
            text_variable.set(filedialog.askdirectory(
                initialdir="",
                title=text
            )),
            **kwargs)


class FilePathButton(Button):
    def __init__(self, text, filetypes, text_variable, **kwargs):
        super().__init__(
            text=text,
            command=lambda:
            text_variable.set(filedialog.askopenfilename(
                initialdir="",
                title=text,
                filetypes=filetypes
            )),
            **kwargs)


class ClearAllButton(Button):
    def __init__(self, variables, **kwargs):
        super().__init__(
            text="Clear All",
            command=lambda: reset(variables),
            **kwargs
        )


def reset(variables):
    for v in variables:
        if isinstance(v, tk.StringVar):
            v.set("")
