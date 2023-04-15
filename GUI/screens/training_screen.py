import tkinter as tk
from tkinter.ttk import *

from GUI.components.path_frame import DirectoryPathFrame
from GUI.components.top_bar import TopBar
from GUI.components.util import SetToDefaultsButton


class TrainingScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='training_screen', **kwargs)

        # top navigation bar
        TopBar(
            master=self,
            controller=controller,
            title="Train New Model",
        ).pack(fill="both")

        # content
        content = Frame(self, name="content")
        content.pack()
        indent = 25
        title_gap = 10
        sub_gap = 0
        # model name
        model_name = tk.StringVar()
        Label(
            master=content,
            text="Model Name:"
        ).pack(
            anchor="w",
            padx=indent,
            pady=title_gap
        )
        Entry(
            master=content,
            width=50,
            textvariable=model_name
        ).pack(
            anchor="w",
            padx=indent*2,
            pady=sub_gap
        )

        # dataset paths
        Label(
            master=content,
            text="Dataset Paths:"
        ).pack(
            anchor="w",
            padx=indent,
            pady=title_gap
        )
        # inputs
        input_path = tk.StringVar()
        DirectoryPathFrame(
            master=content,
            controller=controller,
            text="Inputs Path: ",
            text_variable=input_path
        ).pack(
            anchor="w",
            padx=indent * 2,
            pady=sub_gap
        )
        # labels
        label_path = tk.StringVar()
        DirectoryPathFrame(
            master=content,
            controller=controller,
            text="Labels Path: ",
            text_variable=label_path
        ).pack(
            anchor="w",
            padx=indent * 2,
            pady=sub_gap
        )

        # action buttons
        ButtonFrame(
            master=self,
            controller=controller,
            variables=[
                model_name,
                input_path,
                label_path
            ],
            current_model=current_model
        ).pack(
            side="bottom",
            fill="both"
        )


class ButtonFrame(Frame):
    def __init__(self, master, controller, variables, current_model, **kwargs):
        super().__init__(master, name='button_frame', **kwargs)
        content = Frame(self, name="content")
        content.pack()

        # content
        # back
        Button(
            master=content,
            text="Back",
            command=lambda: controller.navigate("model")
        ).pack(side='left')

        # set to defaults button
        SetToDefaultsButton(
            master=content,
            variables=variables
        ).pack(side='left')

        # train button
        Button(
            master=content,
            text="Train",
            command=lambda: current_model.create_new_model(variables)
        ).pack(side='left')
