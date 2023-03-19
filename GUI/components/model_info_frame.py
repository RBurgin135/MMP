from tkinter.ttk import *
from tkinter import filedialog


class ModelInfoFrame(LabelFrame):
    def __init__(self, master, controller, current_model, use_buttons, **kwargs):
        super().__init__(master, text="info frame", style='Custom.TFrame', **kwargs)

        # content
        # model title label
        ModelTitle(
            master=self,
            current_model=current_model
        ).pack(
            side="top",
            fill="both"
        )

        # model info list
        if current_model.has_data():
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


class ModelTitle(Label):
    def __init__(self, current_model, **kwargs):
        if current_model.has_data():
            super().__init__(
                text=current_model.name,
                **kwargs
            )
        else:
            super().__init__(
                text="No Stored Model",
                **kwargs
            )


class ModelInfo(LabelFrame):
    def __init__(self, master, current_model, **kwargs):
        super().__init__(master, text="model info", **kwargs)

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


class ButtonFrame(LabelFrame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, text="button frame", style='Custom.TFrame', **kwargs)
        content = LabelFrame(self, text="content")
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
            master=content,
            text="New",
            command=lambda: controller.navigate("training")
        ).pack(side='left')

        # save
        Button(
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
            master=content,
            text="Load",
            command=lambda: filedialog.askopenfilename(
                title="Load a model",
                initialdir=initial_dir,
                filetypes=filetypes
            )
        ).pack(side='left')

