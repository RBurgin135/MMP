from tkinter import *

from GUI.components.mode_apply_frame import ModelApplyFrame
from GUI.components.model_info_frame import ModelInfoFrame


class ModelScreen(LabelFrame):
    def __init__(self, master, controller):
        super().__init__(master, text="model screen")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)

        # content
        is_data = False
        m_frame = ModelInfoFrame(self, controller, is_data)
        a_frame = ModelApplyFrame(self, controller, is_data)
        m_frame.grid(
            column=0,
            row=0,
            sticky=N + S + W + E
        )
        a_frame.grid(
            column=1,
            row=0,
            sticky=N + S + W + E
        )
