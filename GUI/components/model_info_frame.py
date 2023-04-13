from tkinter.ttk import *
from tkinter import filedialog


class ModelInfoFrame(Frame):
    def __init__(self, master, controller, current_model, use_buttons, **kwargs):
        super().__init__(master, name='info_frame', style='Custom.TFrame', **kwargs)

        # content
        if not current_model.has_data():
            Label(
                master=self,
                text="No Stored Model"
            ).pack(
                pady=15,
                anchor='n'
            )
        else:
            # model title
            Label(
                master=self,
                text=current_model.name
            ).pack(
                pady=15,
                anchor='n'
            )

            # model info list
            ModelInfo(
                master=self,
                current_model=current_model
            ).pack(
                fill="both"
            )

        # buttons
        if use_buttons:
            ButtonFrame(
                master=self,
                controller=controller,
                current_model=current_model
            ).pack(
                side="bottom",
                fill="both"
            )


class ModelInfo(Frame):
    def __init__(self, master, current_model, **kwargs):
        super().__init__(master, name='model_info', **kwargs)

        # content
        for header_text, text in current_model.get_info():
            # header
            Label(
                master=self,
                text=header_text
            ).pack()
            for t in text:
                Label(
                    master=self,
                    text=t
                ).pack()


class ButtonFrame(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='button_frame', style='Custom.TFrame', **kwargs)
        content = Frame(self, name="content")
        content.pack()

        # filesystem info
        filetypes = (
            ('Tensorflow Files', '*.tf'),
            ('All files', '*.*')
        )
        initial_dir = ""

        # content
        # new
        Button(
            name='new_button',
            master=content,
            text="New",
            command=lambda: controller.navigate("training")
        ).pack(side='left')

        # save
        Button(
            name='save_button',
            master=content,
            text="Save",
            command=lambda: filedialog.asksaveasfilename(
                title="Save a model",
                initialdir=initial_dir,
                filetypes=filetypes
            ),
            state="normal" if current_model.has_data() else"disabled"
        ).pack(side='left')

        # load
        Button(
            name='load_button',
            master=content,
            text="Load",
            command=lambda: filedialog.askopenfilename(
                title="Load a model",
                initialdir=initial_dir,
                filetypes=filetypes
            )
        ).pack(side='left')

