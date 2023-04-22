import tkinter as tk
from tkinter.ttk import *

from GUI.components.single_apply_frame import SingleApplyFrame
from GUI.components.directory_apply_frame import DirectoryApplyFrame
from GUI.components.util import ClearAllButton


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
            text="Apply Model",
            style='Title.TLabel'
        ).pack(
            padx=5,
            pady=15,
            anchor="n"
        )

        # no data image
        image_frame = Frame(self, name='image_frame')
        image_frame.pack(anchor='n', pady=125)
        self.no_data_image = tk.PhotoImage(file="GUI/assets/no_data.png")
        Label(
            name='no_data_image',
            master=image_frame,
            image=self.no_data_image
        ).pack(
            anchor="n",
            padx=5,
            pady=20
        )
        # no data message
        Label(
            master=image_frame,
            text="Load or train a model to begin"
        ).pack(
            anchor="n",
            padx=5,
            pady=10
        )


class ButtonFrame(Frame):
    def __init__(self, master, controller, current_model, variables, command, **kwargs):
        super().__init__(master, name='button_frame', **kwargs)
        content = Frame(self, name="content")
        content.pack()

        # content
        # set to defaults button
        ClearAllButton(
            master=content,
            variables=variables
        ).pack(side='left')

        # begin button
        Button(
            name='begin_button',
            master=content,
            text="Begin",
            command=command
        ).pack(side='left')
