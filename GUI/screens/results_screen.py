from tkinter import filedialog
from tkinter.ttk import *
import cv2
import numpy as np


class ResultsScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='results_screen', **kwargs)
        self.show_image = None

        # results
        Label(
            name='result_image_caption',
            master=self,
            text='Result',
            style='Title.TLabel'
        ).pack(
            padx=5,
            pady=10,
            anchor='n'
        )
        Label(
            name='result_image',
            master=self,
            text=""
        ).pack(
            padx=5,
            pady=20,
            anchor='n'
        )

        # button frame
        ButtonFrame(
            master=self,
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
        self.cv2_image = None
        self.tensor = None

        # content
        # back
        Button(
            name='back_button',
            master=content,
            text="Back",
            command=lambda: controller.navigate('model')
        ).pack(side='left')

        def save_image():
            # file system dialog
            path = filedialog.asksaveasfilename(
                title="Save a model",
                initialdir="",
                filetypes=(
                    ('PNG files', '*.png'),
                    ('JPG files', '*.jpg'),
                    ('All files', '*')
                ),
                defaultextension='.png'
            )
            # save
            cv2.imwrite(path, self.cv2_image)
        # save image
        Button(
            name='save_image_button',
            master=content,
            text="Save Image",
            command=save_image
        ).pack(side='left')

        def save_tensor():
            # file system dialog
            path = filedialog.asksaveasfilename(
                title="Save a model",
                initialdir="",
                filetypes=(
                    ('NumPy files', '*.npy'),
                    ('All files', '*')
                ),
                defaultextension='.npy'
            )
            # save
            np.save(path, self.tensor)
        # save tensor
        Button(
            name='save_tensor_button',
            master=content,
            text="Save Tensor",
            command=save_tensor
        ).pack(side='left')


