from tkinter.ttk import *

from GUI.components.top_bar import TopBar


class ResultScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='result_screen', **kwargs)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # top bar
        TopBar(
            master=self,
            controller=controller,
            title="Results"
        ).grid(
            column=0,
            row=0,
            sticky="new"
        )

        # content
        content = Frame(self, name='content')
        content.grid(
            column=0,
            row=1,
            sticky="nsew"
        )

        # results
        Label(
            name='result_image',
            master=content,
            text="result placeholder"
        ).pack(
            padx=5,
            pady=30,
            anchor='n'
        )
        Label(
            name='result_image_caption',
            master=content,
            text='Result'
        ).pack(
            padx=5,
            pady=10,
            anchor='n'
        )

        # button frame
        ButtonFrame(
            master=content,
            controller=controller,
            current_model=current_model
        ).pack(
            side='bottom',
            fill='both'
        )


class ButtonFrame(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='button_frame', **kwargs)
        content = Frame(self, name='content')
        content.pack()

        # content
        # back
        Button(
            name='back_button',
            master=content,
            text="Back",
            command=lambda: controller.navigate('model')
        ).pack(side='left')

        # redo
        Button(
            name='redo_button',
            master=content,
            text="Redo",
            command=lambda: print(0)
        ).pack(side='left')

        # save
        Button(
            name='save_button',
            master=content,
            text="Save",
            command=lambda: print(0)
        ).pack(side='left')


