import tkinter as tk
from tkinter.ttk import *

from GUI.components.path_frame import PathFrame


class ModelApplyFrame(LabelFrame):
    def __init__(self, master, controller, data, *args, **kwargs):
        super().__init__(master, text="apply frame", *args, **kwargs)

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
        PathFrame(
            master=content,
            controller=controller,
            text="Dataset Path: "
        ).pack(
            padx=10,
            pady=3,
            anchor="w"
        )

        # output path button
        PathFrame(
            master=content,
            controller=controller,
            text="Output Path: "
        ).pack(
            padx=10,
            pady=3,
            anchor="w"
        )

        # output type button
        output_type_button(
            master=content
        ).pack(
            padx=10,
            pady=3,
            anchor="w"
        )

        # action button frame
        ActionButtonFrame(
            master=self,
            controller=controller
        ).pack(
            side="bottom",
            fill="both"
        )


class ActionButtonFrame(LabelFrame):
    def __init__(self, master, controller, *args, **kwargs):
        super().__init__(master, text="action button frame", *args, **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # content
        # set to defaults button
        Button(
            master=self,
            text="Set to Defaults",
            command=lambda: print("defaults")
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


def output_type_button(master):
    description = "Output type"
    options = [
        description,
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
