import tkinter as tk
from tkinter.ttk import *

from GUI.components.path_frame import PathFrame


class DirectoryApplyFrame(Frame):
    def __init__(self, master, controller, current_model, button_frame, **kwargs):
        super().__init__(master, name='apply_frame', **kwargs)

        # content
        # title
        Label(
            master=self,
            text="Apply Model to Directory"
        ).pack(
            padx=5,
            pady=15,
            anchor="n"
        )

        # dataset button
        dataset_path = tk.StringVar()
        PathFrame(
            master=self,
            controller=controller,
            text="Dataset Path: ",
            text_variable=dataset_path
        ).pack(
            padx=10,
            pady=3,
            anchor="w"
        )

        # output path button
        output_path = tk.StringVar()
        PathFrame(
            master=self,
            controller=controller,
            text="Output Path: ",
            text_variable=output_path
        ).pack(
            padx=10,
            pady=3,
            anchor="w"
        )

        # action button frame
        button_frame(
            master=self,
            controller=controller,
            current_model=current_model,
            variables=[
                dataset_path,
                output_path
            ]
        ).pack(
            side="bottom",
            fill="both"
        )
