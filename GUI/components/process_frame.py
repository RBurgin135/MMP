from tkinter.ttk import *
from tkinter import messagebox


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
        super().__init__(master, text="button frame", **kwargs)
        content = LabelFrame(self, text="content")
        content.pack()

        # content
        # abort
        Button(
            master=content,
            text="Abort",
            command=lambda: abort_dialog(controller)

        ).pack(side='left')

        # done
        Button(
            master=content,
            text="Done",
            command=lambda: print("done"),
            state='disabled'
        ).pack(side='left')


def abort_dialog(controller):
    if messagebox.askyesno(
            title="Confirm Abort",
            message="Are you sure you want to abort the process? All progress will be lost"):
        controller.navigate("model")
