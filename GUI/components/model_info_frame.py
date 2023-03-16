from tkinter.ttk import *


class ModelInfoFrame(LabelFrame):
    def __init__(self, master, controller, data):
        super().__init__(master, text="info frame", style='Custom.TFrame')
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        # content
        is_data = True

        # model title label
        ModelTitle(
            master=self,
            is_data=is_data
        ).grid(
            column=0,
            row=0,
            sticky="n"
        )

        # model info list
        ModelInfo(
            master=self,
            is_data=is_data
        ).grid(
            column=0,
            row=1,
            sticky="nw"
        )


class ModelTitle(Label):
    def __init__(self, is_data, *args, **kwargs):
        if is_data:
            super().__init__(
                text="Model Name",
                *args,
                **kwargs
            )
        else:
            super().__init__(
                text="No Stored Model",
                *args,
                **kwargs
            )


class ModelInfo(Label):
    def __init__(self, is_data, *args, **kwargs):
        if not is_data:
            super().__init__(
                *args,
                **kwargs
            )
        else:
            super().__init__(
                text="info",
                *args,
                **kwargs
            )
