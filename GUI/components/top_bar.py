from tkinter.ttk import *


class TopBar(Frame):
    def __init__(self, master, controller, title, **kwargs):
        super().__init__(master, name='top_navigation_bar', **kwargs)

        # content
        # title
        Label(
            master=self,
            text=title
        ).grid(
            column=1,
            row=0
        )
