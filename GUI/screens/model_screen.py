from tkinter import *

from GUI.components.model_apply_notebook import ModelApplyNotebook, NoDataFrame
from GUI.components.model_info_frame import ModelInfoFrame


class ModelScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='model_screen', **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)

        # content
        # model information frame
        ModelInfoFrame(
            master=self,
            controller=controller,
            current_model=current_model,
            use_buttons=True
        ).grid(
            column=0,
            row=0,
            sticky="nsew"
        )

        # model application frame
        if current_model.pca_wavelet_model is not None:
            ModelApplyNotebook(
                master=self,
                controller=controller,
                current_model=current_model
            ).grid(
                column=1,
                row=0,
                sticky="nsew"
            )
        else:
            NoDataFrame(
                master=self
            ).grid(
                column=1,
                row=0,
                sticky="nsew"
            )
