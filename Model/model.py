import threading
from tkinter import filedialog

import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageTk, Image

from Model import processes
from PCA_Wavelet_Codebase.custom_layers.conv_2d_transpose_seperable_layer import Conv2DTransposeSeparableLayer
from PCA_Wavelet_Codebase.custom_layers.mean_layer import MeanLayer
from PCA_Wavelet_Codebase.custom_layers.symmetric_padding_2d import SymmetricPadding2D


class Model:
    def __init__(self, controller):
        self.name = None
        self.controller = controller
        self.pca_wavelet_model = None

        # filesystem info
        self.filetypes = (
            ('HDF5 files', '*.h5'),
            ('All files', '*')
        )
        self.initial_dir = ""

    def create_new_model(self, variables):
        # extract from variables
        self.name = variables[0].get()
        images_path = variables[1].get()
        labels_path = variables[2].get()

        # build
        thread = processes.Build(
            current_model=self,
            images_path=images_path,
            labels_path=labels_path
        )
        thread.start()

        # navigate
        self.controller.navigate("process")
        self.give_process_info(
            title="Training",
            process=thread
        )

    def apply_model_to_dir(self, variables):
        # extract from variables
        images_path = variables[0].get() + '/'
        output_path = variables[1].get() + '/'

        # apply
        thread = processes.ApplyToDir(
            current_model=self,
            images_path=images_path,
            output_path=output_path
        )
        thread.start()

        # navigate
        self.controller.navigate("process")
        self.give_process_info(
            title="Applying to Directory",
            process=thread
        )

    def apply_model_to_image(self, variables):
        # extract from variables
        image_path = variables[0].get()

        # apply
        image = cv2.imread(image_path)
        prediction = self.pca_wavelet_model(np.reshape(image, (1, 64, 64, 3)))

        # navigate
        self.controller.navigate('result')

        # show result image
        self.give_results_prediction(prediction)

    def save_model(self):
        # file system dialog
        path = filedialog.asksaveasfilename(
            title="Save a model",
            initialdir=self.initial_dir,
            filetypes=self.filetypes
        )

        # error checking
        if path == "" or path is None:
            return

        # add file type
        if path.split('.')[-1] != 'h5':
            path += '.h5'

        # multithread save
        def save():
            self.pca_wavelet_model.compile(optimizer='adam', loss='categorical_crossentropy')
            self.pca_wavelet_model.save(path)
            self.controller.navigate('model')

        # start process
        self.controller.navigate('saving')
        thread = threading.Thread(target=save)
        thread.start()

    def load_model(self):
        # file system dialog
        path = filedialog.askopenfilename(
            title="Load a model",
            initialdir=self.initial_dir,
            filetypes=self.filetypes
        )

        # multithread load
        def load():
            try:
                self.pca_wavelet_model = tf.keras.models.load_model(
                    path,
                    custom_objects={
                        'Conv2DTransposeSeparableLayer': Conv2DTransposeSeparableLayer,
                        'MeanLayer': MeanLayer,
                        'SymmetricPadding2D': SymmetricPadding2D
                    }
                )
            except IOError:
                pass
            self.controller.navigate('model')

        # start process
        self.controller.navigate('loading')
        thread = threading.Thread(target=load)
        thread.start()

    def give_process_info(self, title, process):
        # set title
        process_screen = self.controller.children['process_screen']
        process_screen.children['title'].configure(text=title)

        # set process
        button_frame = process_screen.children['content']\
            .children['process_frame'].children['button_frame']
        button_frame.process = process

    def give_results_prediction(self, prediction):
        # get references
        results_screen = self.controller.children['results_screen']
        button_frame = results_screen.children['button_frame']

        # take images
        button_frame.cv2_image = np.array(prediction[0, :, :, 1] * 255)
        image = Image.fromarray(button_frame.cv2_image)
        results_screen.show_image = ImageTk.PhotoImage(
            image=image.resize((200, 200), Image.NEAREST)
        )

        # reconfigure result image
        results_screen.children['result_image'].configure(image=results_screen.show_image)

    def has_data(self):
        return self.pca_wavelet_model is not None

    def get_info(self):
        return [
            ("Name: ",
             [self.name]
             )
        ]
