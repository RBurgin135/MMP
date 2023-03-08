from tkinter import *

from GUI.components.model_apply_frame import ModelApplyFrame
from GUI.components.model_info_frame import ModelInfoFrame


class ModelScreen(LabelFrame):
    def __init__(self, master, controller, *args, **kwargs):
        super().__init__(master, text="model screen", *args, **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)

        # content
        is_data = False

        # model information frame
        ModelInfoFrame(
            master=self,
            controller=controller,
            data=is_data
        ).grid(
            column=0,
            row=0,
            sticky="nsew"
        )

        # model application frame
        ModelApplyFrame(
            master=self,
            controller=controller,
            data=is_data
        ).grid(
            column=1,
            row=0,
            sticky="nsew"
        )
