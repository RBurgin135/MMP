from tkinter.ttk import *


class ButtonFrame(LabelFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, text="button frame", **kwargs)
        # content
        self.content = LabelFrame(self, text="content")
        self.content.pack()

    def add_buttons(self, buttons):
        # buttons
        i = 0
        self.content.rowconfigure(0, weight=1)
        for b in buttons:
            b.grid(column=i, row=0)
            self.content.columnconfigure(i, weight=1)
            i += 1
