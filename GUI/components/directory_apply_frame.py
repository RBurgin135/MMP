import tkinter as tk
from tkinter.ttk import *

from GUI.components.path_frame import DirectoryPathFrame


class DirectoryApplyFrame(Frame):
    def __init__(self, master, controller, current_model, button_frame, **kwargs):
        super().__init__(master, name='apply_frame', **kwargs)

        # content
        # title
        Label(
            master=self,
            text="Apply Model to Directory",
            style='Title.TLabel'
        ).pack(
            padx=5,
            pady=15,
            anchor="n"
        )

        # dataset path button
        dataset_path = tk.StringVar()
        DirectoryPathFrame(
            master=self,
            controller=controller,
            text="Dataset Path: ",
            text_variable=dataset_path
        ).pack(
            padx=10,
            pady=3,
            anchor="n"
        )

        # output path button
        output_path = tk.StringVar()
        DirectoryPathFrame(
            master=self,
            controller=controller,
            text="Output Path: ",
            text_variable=output_path
        ).pack(
            padx=10,
            pady=3,
            anchor="n"
        )

        # action button frame
        variables = [dataset_path, output_path]
        button_frame(
            master=self,
            controller=controller,
            current_model=current_model,
            variables=variables,
            command=lambda: current_model.apply_model_to_dir(variables)
        ).pack(
            side="bottom",
            fill="both"
        )
