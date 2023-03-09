from tkinter.ttk import *

from GUI.components.model_info_frame import ModelInfoFrame
from GUI.components.process_frame import ProcessFrame
from GUI.components.top_bar import TopBar


class ProcessScreen(LabelFrame):
    def __init__(self, master, controller, *args, **kwargs):
        super().__init__(master, text="process screen", *args, **kwargs)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # top bar
        TopBar(
            master=self,
            controller=controller,
            title="Process"
        ).grid(
            column=0,
            row=0,
            sticky="new"
        )

        # content
        content = LabelFrame(self, text="content")
        content.grid(
            column=0,
            row=1,
            sticky="nsew"
        )
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=5)
        is_data = False

        # model information frame
        ModelInfoFrame(
            master=content,
            controller=controller,
            data=is_data
        ).grid(
            column=0,
            row=0,
            sticky="nsew"
        )

        # process frame
        ProcessFrame(
            master=content,
            controller=controller
        ).grid(
            column=1,
            row=0,
            sticky="nsew"
        )
