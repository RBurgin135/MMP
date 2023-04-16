from tkinter import filedialog
from tkinter.ttk import *
import cv2
import numpy as np
from PIL import ImageTk, Image


class ResultsScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='results_screen', **kwargs)
        self.show_image = None

        # results
        Label(
            name='result_image_caption',
            master=self,
            text='Result'
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

    def take_info(self, variables, prediction):
        button_frame = self.children['button_frame']

        # take images
        button_frame.cv2_image = np.array(prediction[0, :, :, 1] * 255)
        image = Image.fromarray(button_frame.cv2_image)
        self.show_image = ImageTk.PhotoImage(
            image=image.resize((200, 200), Image.NEAREST)
        )

        # reconfigure result image
        self.children['result_image'].configure(image=self.show_image)


class ButtonFrame(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, name='button_frame', **kwargs)
        content = Frame(self, name='content')
        content.pack()
        self.cv2_image = None

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
                )
            )
            # save
            cv2.imwrite(path, self.cv2_image)
        # save
        Button(
            name='save_button',
            master=content,
            text="Save",
            command=save_image
        ).pack(side='left')


