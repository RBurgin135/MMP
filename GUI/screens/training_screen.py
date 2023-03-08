from tkinter import PhotoImage
from tkinter.ttk import *
from PIL import ImageTk

from GUI.components.path_frame import PathFrame
from GUI.components.top_nav_bar import TopNavigationBar, NavButtonData


class TrainingScreen(LabelFrame):
    def __init__(self, master, controller, *args, **kwargs):
        super().__init__(master, text="training screen", *args, **kwargs)

        # top navigation bar
        TopNavigationBar(
            master=self,
            controller=controller,
            title="Train New Model",
            nav_button=NavButtonData(
                text="Back",
                command=lambda: controller.navigate("model")
            )
        ).pack(fill="both")

        # content
        content = LabelFrame(self, text="content")
        content.pack()
        indent = 25
        title_gap = 10
        sub_gap = 0
        # model name
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
            width=50
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
        PathFrame(
            master=content,
            controller=controller,
            text="Inputs Path: "
        ).pack(
            anchor="w",
            padx=indent * 2,
            pady=sub_gap
        )
        # labels
        PathFrame(
            master=content,
            controller=controller,
            text="Labels Path: "
        ).pack(
            anchor="w",
            padx=indent * 2,
            pady=sub_gap
        )

        # action buttons
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
            text="Train",
            command=lambda: print("train")
        ).grid(
            column=1,
            row=0,
            sticky="nw"
        )
