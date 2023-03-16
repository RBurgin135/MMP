from tkinter.ttk import *


class ProcessFrame(LabelFrame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, text="process frame", **kwargs)

        # content
        Button(
            master=self,
            text="Abort",
            command=lambda: controller.navigate("model")
        ).pack()
