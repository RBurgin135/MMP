from tkinter.ttk import *

from GUI.components.model_info_frame import ModelInfoFrame
from GUI.components.process_frame import ProcessFrame


class ProcessScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='process_screen', **kwargs)

        # model information frame
        ModelInfoFrame(
            master=self,
            controller=controller,
            current_model=current_model,
            use_buttons=False
        ).pack(
            side='left',
            fill='both'
        )

        # process frame
        ProcessFrame(
            master=self,
            controller=controller
        ).pack(
            side='left',
            fill='both',
            expand=True
        )
