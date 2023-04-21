from tkinter.ttk import *
from tkinter import messagebox


class ProcessFrame(Frame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, name='process_frame', **kwargs)

        # title
        Label(
            name='title',
            master=self,
            text="Process",
            style='Title.TLabel'
        ).pack(
            pady=10,
            anchor="n"
        )

        # console
        #Console.initialise_widget(self).pack(
        #    pady=10,
        #    anchor='n'
        #)

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
        self.controller = controller
        self.process = None
        content = Frame(self, name='content')
        content.pack()

        # content
        # abort
        Button(
            master=content,
            text="Abort",
            command=self.abort_dialog,
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

    def abort_dialog(self):
        if messagebox.askyesno(
                title="Confirm Abort",
                message="Are you sure you want to abort the process? All progress will be lost",
                default=messagebox.NO):
            self.process.abort()
