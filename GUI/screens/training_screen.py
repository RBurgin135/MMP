import tkinter as tk
from tkinter.ttk import *

from GUI.components.path_frame import DirectoryPathFrame
from GUI.components.util import ClearAllButton


class TrainingScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='training_screen', **kwargs)

        # top navigation bar
        Label(
            master=self,
            text="Train New Model",
            style='Title.TLabel'
        ).pack(
            pady=15,
            anchor='n'
        )

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
            width=25,
            textvariable=model_name
        ).pack(
            anchor="w",
            padx=indent * 2,
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
        input_path = tk.StringVar(value='C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill/images')
        DirectoryPathFrame(
            master=content,
            controller=controller,
            text="Inputs Path: ",
            text_variable=input_path
        ).pack(
            anchor="n",
            padx=indent * 2,
            pady=sub_gap
        )
        # labels
        label_path = tk.StringVar(value='C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill/labels')
        DirectoryPathFrame(
            master=content,
            controller=controller,
            text="Labels Path: ",
            text_variable=label_path
        ).pack(
            anchor="n",
            padx=indent * 2,
            pady=sub_gap
        )

        # count
        count = tk.StringVar(value=str(1))
        Label(
            master=content,
            text="Amount from dataset:"
        ).pack(
            anchor="w",
            padx=indent,
            pady=title_gap
        )
        Entry(
            master=content,
            width=10,
            textvariable=count,
            validatecommand=controller.register(lambda char: char.isdigit())
        ).pack(
            anchor="w",
            padx=indent * 2,
            pady=sub_gap
        )

        # layers
        layers = tk.StringVar(value=str(1))
        Label(
            master=content,
            text="Number of layers:"
        ).pack(
            anchor="w",
            padx=indent,
            pady=title_gap
        )
        Entry(
            master=content,
            width=10,
            textvariable=layers,
            validatecommand=controller.register(lambda char: char.isdigit())
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
                label_path,
                count,
                layers
            ],
            current_model=current_model
        ).pack(
            side="bottom",
            fill="both"
        )

        # error label
        Label(
            name='error_label',
            master=self,
            text=""
        ).pack(
            side='bottom'
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
        ClearAllButton(
            master=content,
            variables=variables
        ).pack(side='left')

        # train button
        def train():
            try:
                current_model.create_new_model(variables)
            except ValueError:
                master.children['error_label'].configure(text='Invalid Inputs')

        Button(
            master=content,
            text="Train",
            command=train
        ).pack(side='left')
