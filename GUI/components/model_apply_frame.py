import tkinter as tk
from tkinter.ttk import *
from tkinter import filedialog


class ModelApplyFrame(LabelFrame):
    def __init__(self, master, controller, data):
        super().__init__(master, text="apply frame")

        # content
        # title
        Label(
            master=self,
            text="Apply Model"
        ).pack(
            padx=10,
            pady=10
        )

        # dataset button
        file_path_button(
            master=self,
            text="Dataset path"
        ).pack()

        # output path button
        file_path_button(
            master=self,
            text="Output path"
        ).pack()

        # output type button
        output_type_button(
            master=self
        ).pack()


def file_path_button(master, text):
    return Button(
        master=master,
        text=text,
        width=100,
        command=lambda: filedialog.askopenfilename(
            initialdir="/",
            title=text,
            filetypes=(("python files", "*.py"), ("all files", "*.*"))
        )
    )


def output_type_button(master):
    options = [
        "Output type",
        "Segmentation mask",
        "Point map",
        "etc"
    ]
    clicked = tk.StringVar()
    clicked.set("Segmentation mask")
    return OptionMenu(
        master,
        clicked,
        *options,
        command=lambda x: print(x)
    )
