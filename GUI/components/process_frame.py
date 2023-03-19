from tkinter.ttk import *


class ProcessFrame(LabelFrame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, text="process frame", **kwargs)

        # content
        ButtonFrame(
            master=self,
            controller=controller
        ).pack(
            side="bottom",
            fill="both"
        )


class ButtonFrame(LabelFrame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, text="button frame")
        content = LabelFrame(self, text="content")
        content.pack()

        # content
        # abort
        Button(
            master=content,
            text="Abort",
            command=lambda: controller.navigate("model")
        ).pack(side='left')

        # done
        Button(
            master=content,
            text="Done",
            command=lambda: print("done"),
            state='disabled'
        ).pack(side='left')
