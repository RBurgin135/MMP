from tkinter.ttk import *


class ModelInfoFrame(Frame):
    def __init__(self, master, controller, current_model, use_buttons, **kwargs):
        super().__init__(master, name='info_frame', style='ShadedFrame.TFrame', **kwargs)

        # content
        if not current_model.has_data():
            Label(
                master=self,
                text="No Stored Model",
                style='ShadedTitle.TLabel'
            ).pack(
                pady=15,
                anchor='n'
            )
        else:
            # model title
            Label(
                master=self,
                text=current_model.name,
                style='ShadedTitle.TLabel'
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
        ButtonFrame(
            master=self,
            controller=controller,
            current_model=current_model,
            style='ShadedButton.TButton',
            use_buttons=use_buttons
        ).pack(
            side="bottom",
            fill="both"
        )


class ModelInfo(Frame):
    def __init__(self, master, current_model, **kwargs):
        super().__init__(master, name='model_info', style='ShadedFrame.TFrame', **kwargs)

        # content
        for header_text, text in current_model.get_info():
            # header
            Label(
                master=self,
                text=header_text,
                style='ShadedTitle.TLabel'
            ).pack(
                anchor='w'
            )
            for t in text:
                Label(
                    master=self,
                    text=t,
                    style='ShadedText.TLabel'
                ).pack(
                    anchor='w',
                    padx=10
                )


class ButtonFrame(Frame):
    def __init__(self, master, controller, current_model, style, use_buttons, **kwargs):
        super().__init__(master, name='button_frame', style='ShadedFrame.TFrame', **kwargs)
        content = Frame(self, name="content")
        content.pack()

        # content
        # new
        Button(
            name='new_button',
            master=content,
            text="New",
            command=lambda: controller.navigate("training"),
            state='normal' if use_buttons else 'disabled',
            style=style
        ).pack(side='left')

        # save
        Button(
            name='save_button',
            master=content,
            text="Save",
            command=current_model.save_model,
            state='normal' if use_buttons and current_model.has_data() else 'disabled',
            style=style
        ).pack(side='left')

        # load
        Button(
            name='load_button',
            master=content,
            text="Load",
            command=current_model.load_model,
            state='normal' if use_buttons else 'disabled',
            style=style
        ).pack(side='left')
