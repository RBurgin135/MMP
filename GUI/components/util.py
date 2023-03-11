from tkinter import filedialog
from tkinter.ttk import *


class DirectoryPathButton(Button):
    def __init__(self, text, text_variable, *args, **kwargs):
        super().__init__(
            text=text,
            command=lambda:
            text_variable.set(filedialog.askdirectory(
                initialdir="/",
                title=text
            ))
            , *args, **kwargs)


class FilePathButton(Button):
    def __init__(self, text, filetypes, text_variable, *args, **kwargs):
        super().__init__(
            text=text,
            command=lambda:
            text_variable.set(filedialog.askopenfilename(
                initialdir="/",
                title=text,
                filetypes=filetypes
            ))
            , *args, **kwargs)
