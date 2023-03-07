from tkinter.ttk import *
from GUI.screens.training_screen import TrainingScreen


class ModelInfoFrame(LabelFrame):
    def __init__(self, master, controller, data):
        super().__init__(master, text="info frame")
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        # content
        is_data = True

        # model title label
        model_title(
            master=self,
            is_data=is_data
        ).grid(
            column=0,
            row=0,
            sticky="n"
        )

        # model info list
        model_info(
            master=self,
            is_data=is_data
        ).grid(
            column=0,
            row=1,
            sticky="nw"
        )

        # button frame
        ButtonFrame(
            master=self,
            controller=controller
        ).grid(
            column=0,
            row=2,
            sticky="ews"
        )


class ButtonFrame(LabelFrame):
    def __init__(self, master, controller):
        super().__init__(master, text="button frame")

        # content
        new = Button(
            master=self,
            text="New",
            command=lambda: controller.show_frame(TrainingScreen)
        )
        save = Button(
            master=self,
            text="Save",
            command=lambda: print("save")
        )
        load = Button(
            master=self,
            text="Load",
            command=lambda: print("load")
        )
        new.grid(column=0, row=0, padx=5, sticky="wen")
        save.grid(column=1, row=0, padx=5, sticky="wen")
        load.grid(column=2, row=0, padx=5, sticky="wen")
        self.rowconfigure(0, weight=1)
        for c in range(3):
            self.columnconfigure(c, weight=1)


def model_title(master, is_data):
    if is_data:
        return Label(
            master=master,
            text="Model Name"
        )
    else:
        return Label(
            master=master,
            text="No Stored Model",
        )


def model_info(master, is_data):
    if not is_data:
        return Label(master=master)
    return Label(
        master=master,
        text="info"
    )
