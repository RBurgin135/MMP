from tkinter import filedialog
from tkinter.ttk import *


class FilePathButton(Button):
    def __init__(self, text, filetypes, *args, **kwargs):
        super().__init__(
            text=text,
            command=lambda: filedialog.askopenfilename(
                initialdir="/",
                title=text,
                filetypes=filetypes
            ), *args, **kwargs)
