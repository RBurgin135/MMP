import tkinter as tk
from tkinter.ttk import *

from PIL import Image, ImageTk

from GUI.components.path_frame import FilePathFrame


class SingleApplyFrame(Frame):
    def __init__(self, master, controller, current_model, button_frame):
        super().__init__(master, name='single_apply_frame')

        # title
        Label(
            master=self,
            text="Apply Model to Image"
        ).pack(
            padx=5,
            pady=15,
            anchor="n"
        )

        # file path frame
        image_path = tk.StringVar()
        FilePathFrame(
            master=self,
            controller=controller,
            text="Image Path: ",
            text_variable=image_path
        ).pack(
            padx=10,
            pady=3,
            anchor="w"
        )

        # load preview
        def load_preview():
            preview_button = controller.children['model_screen'].children['single_apply_frame'].children['preview']
            try:
                image = Image.open(image_path.get())
                self.image = ImageTk.PhotoImage(image.resize((200, 200)))
                preview_button.configure(image=self.image)
            except (FileNotFoundError, AttributeError):
                preview_button.configure(text="File not found")
                self.image = None

        Button(
            name='preview_label',
            master=self,
            text="Preview",
            command=load_preview
        ).pack(
            padx=10,
            pady=3,
            anchor='n'
        )

        # preview
        Label(
            name='preview',
            master=self,
            text=""
        ).pack(
            padx=10,
            pady=15,
            anchor='n'
        )

        # action button frame
        variables = [image_path]
        button_frame(
            master=self,
            controller=controller,
            current_model=current_model,
            variables=variables,
            command=lambda: current_model.apply_model_to_image(variables)
        ).pack(
            side="bottom",
            fill="both"
        )
