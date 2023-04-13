from tkinter import filedialog
from tkinter.ttk import *


class DirectoryPathButton(Button):
    def __init__(self, text, text_variable, *args, **kwargs):
        super().__init__(
            text=text,
            command=lambda:
            text_variable.set(filedialog.askdirectory(
                initialdir="",
                title=text
            ))
            , *args, **kwargs)


class FilePathButton(Button):
    def __init__(self, text, filetypes, text_variable, *args, **kwargs):
        super().__init__(
            text=text,
            command=lambda:
            text_variable.set(filedialog.askopenfilename(
                initialdir="",
                title=text,
                filetypes=filetypes
            ))
            , *args, **kwargs)


class SetToDefaultsButton(Button):
    def __init__(self, variables, *args, **kwargs):
        super().__init__(
            text="Set to Defaults",
            command=lambda: reset(variables),
            *args,
            **kwargs
        )


def reset(variables):
    for v in variables:
        v.set("")
