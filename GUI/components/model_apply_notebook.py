import tkinter as tk
from tkinter.ttk import *

from GUI.components.single_apply_frame import SingleApplyFrame
from GUI.components.directory_apply_frame import DirectoryApplyFrame
from GUI.components.util import SetToDefaultsButton


class ModelApplyNotebook(Notebook):
    def __init__(self, master, controller, current_model):
        super().__init__(master, name='apply_notebook')

        # single apply tab
        self.add(
            SingleApplyFrame(
                master=master,
                controller=controller,
                current_model=current_model,
                button_frame=ButtonFrame
            ),
            text="Apply to Image"
        )

        # directory apply tab
        self.add(
            DirectoryApplyFrame(
                master=master,
                controller=controller,
                current_model=current_model,
                button_frame=ButtonFrame
            ),
            text="Apply to Directory"
        )


class NoDataFrame(Frame):
    def __init__(self, master):
        super().__init__(master)

        # title
        Label(
            master=self,
            text="Apply Model"
        ).pack(
            padx=5,
            pady=15,
            anchor="n"
        )

        # no data image
        self.no_data_image = tk.PhotoImage(file="GUI/assets/no_data.png")
        Label(
            name='no_data_image',
            master=self,
            image=self.no_data_image,
            width=1
        ).pack(
            anchor="n",
            padx=5,
            pady=10
        )
        # no data message
        Label(
            master=self,
            text="Load or train a model to begin"
        ).pack(
            anchor="n",
            padx=5,
            pady=10
        )


class ButtonFrame(Frame):
    def __init__(self, master, controller, current_model, variables, **kwargs):
        super().__init__(master, name='button_frame', **kwargs)
        content = Frame(self, name="content")
        content.pack()

        # content
        # set to defaults button
        SetToDefaultsButton(
            master=content,
            variables=variables
        ).pack(side='left')

        # begin button
        Button(
            name='begin_button',
            master=content,
            text="Begin",
            command=lambda: current_model.apply_model(variables)
        ).pack(side='left')


class OutputTypeFrame(Frame):
    def __init__(self, text_variable, **kwargs):
        super().__init__(**kwargs)

        # options
        options = [
            "",
            "Segmentation mask",
            "Point map",
            "etc"
        ]
        text_variable.set(options[0])

        # content
        # label
        Label(
            master=self,
            text="Output Type:"
        ).pack(side='left')
        # option menu
        OptionMenu(
            self,
            text_variable,
            *options,
            command=lambda x: print(x)
        ).pack(side='left')
