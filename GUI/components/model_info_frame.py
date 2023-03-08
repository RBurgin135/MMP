from tkinter.ttk import *


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
