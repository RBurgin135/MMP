from tkinter.ttk import *
from tkinter import messagebox


class ProcessFrame(Frame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, name='process_frame', **kwargs)

        # content
        ButtonFrame(
            master=self,
            controller=controller
        ).pack(
            side="bottom",
            fill="both"
        )


class ButtonFrame(Frame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, name='button_frame', **kwargs)
        content = Frame(self, name='content')
        content.pack()

        # content
        # abort
        Button(
            master=content,
            text="Abort",
            command=lambda: abort_dialog(controller),
            name='abort_button'
        ).pack(side='left')

        # done
        Button(
            master=content,
            text="Done",
            command=lambda: controller.navigate('model'),
            state='disabled',
            name='done_button'
        ).pack(side='left')


def abort_dialog(controller):
    if messagebox.askyesno(
            title="Confirm Abort",
            message="Are you sure you want to abort the process? All progress will be lost"):
        controller.navigate("model")
