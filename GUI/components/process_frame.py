from tkinter.ttk import *


class ProcessFrame(LabelFrame):
    def __init__(self, master, controller, *args, **kwargs):
        super().__init__(master, text="process frame", *args, **kwargs)

        # content
        Button(
            master=self,
            text="Abort",
            command=lambda: controller.navigate("model")
        ).pack()
