from tkinter.ttk import *

from GUI.components.model_info_frame import ModelInfoFrame
from GUI.components.process_frame import ProcessFrame


class ProcessScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='process_screen', **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)

        # model information frame
        ModelInfoFrame(
            master=self,
            controller=controller,
            current_model=current_model,
            use_buttons=False
        ).grid(
            column=0,
            row=0,
            sticky="nsew"
        )

        # process frame
        ProcessFrame(
            master=self,
            controller=controller
        ).grid(
            column=1,
            row=0,
            sticky="nsew"
        )
