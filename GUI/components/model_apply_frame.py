import tkinter as tk
from tkinter.ttk import *

from GUI.components.path_frame import PathFrame
from GUI.components.util import SetToDefaultsButton


class ModelApplyFrame(LabelFrame):
    def __init__(self, master, controller, data, **kwargs):
        super().__init__(master, text="apply frame", **kwargs)

        # content
        # title
        content = LabelFrame(master=self, text="content")
        content.pack(fill="both")
        Label(
            master=content,
            text="Apply Model"
        ).pack(
            padx=5,
            pady=10,
            anchor="w"
        )

        # dataset button
        dataset_path = tk.StringVar()
        PathFrame(
            master=content,
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
            master=content,
            controller=controller,
            text="Output Path: ",
            text_variable=output_path
        ).pack(
            padx=10,
            pady=3,
            anchor="w"
        )

        # output type button
        output_type = tk.StringVar()
        OutputTypeFrame(
            master=content,
            text_variable=output_type
        ).pack(
            padx=10,
            pady=3,
            anchor="w"
        )

        # action button frame
        ActionButtonFrame(
            master=self,
            controller=controller,
            variables=[
                dataset_path,
                output_path,
                output_type
            ]
        ).pack(
            side="bottom",
            fill="both"
        )


class ActionButtonFrame(LabelFrame):
    def __init__(self, master, controller, variables, **kwargs):
        super().__init__(master, text="action button frame", **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # content
        # set to defaults button
        SetToDefaultsButton(
            master=self,
            variables=variables
        ).grid(
            column=0,
            row=0,
            sticky="ne"
        )

        # train button
        Button(
            master=self,
            text="Begin",
            command=lambda: controller.navigate("process")
        ).grid(
            column=1,
            row=0,
            sticky="nw"
        )


class OutputTypeFrame(Frame):
    def __init__(self, text_variable, **kwargs):
        super().__init__(**kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
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
        ).grid(
            column=0,
            row=0
        )
        # option menu
        OptionMenu(
            self,
            text_variable,
            *options,
            command=lambda x: print(x)
        ).grid(
            column=1,
            row=0
        )

