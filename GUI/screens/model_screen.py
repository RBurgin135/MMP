from tkinter import *

from GUI.components.model_apply_notebook import ModelApplyNotebook, NoDataFrame
from GUI.components.model_info_frame import ModelInfoFrame


class ModelScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='model_screen', **kwargs)

        # content
        # model information frame
        ModelInfoFrame(
            master=self,
            controller=controller,
            current_model=current_model,
            use_buttons=True
        ).pack(
            side='left',
            fill='both'
        )

        # model application frame
        if current_model.has_data():
            ModelApplyNotebook(
                master=self,
                controller=controller,
                current_model=current_model
            ).pack(
                side='left',
                fill='both',
                expand=True
            )
        else:
            NoDataFrame(
                master=self
            ).pack(
                side='left',
                fill='both',
                expand=True
            )
