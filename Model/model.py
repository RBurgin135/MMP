import os
import threading
from tkinter import filedialog

import cv2
import numpy as np
import tensorflow as tf

from Model.dataset import create_dataset
from PCA_Wavelet_Codebase.build import build_model
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

        # navigate
        self.controller.navigate("process")
        self.give_process_title("Training")
        self.configure_process_screen_buttons(process_done=False)

        # build multithread
        def build():
            dataset = create_dataset(images_path, labels_path)
            head, invhead = build_model(dataset)
            self.pca_wavelet_model = head

            # configure buttons
            self.configure_process_screen_buttons(process_done=True)

        # build
        thread = threading.Thread(target=build)
        thread.start()

    def apply_model_to_dir(self, variables):
        # extract from variables
        images_path = variables[0].get() + '/'
        output_path = variables[1].get() + '/'

        # navigate
        self.controller.navigate("process")
        self.give_process_title("Applying to Directory")
        self.configure_process_screen_buttons(process_done=False)

        # apply model subroutine
        def apply():
            # iterate over images
            for x in os.listdir(images_path):
                # apply
                image = cv2.imread(images_path + x)
                prediction = self.pca_wavelet_model(np.reshape(image, (1, 64, 64, 3)))

                # save
                cv2.imwrite(output_path + x, np.array(prediction[0, :, :, 1] * 255))
                print(f"finished: {x}")

            # configure buttons
            self.configure_process_screen_buttons(process_done=True)

        # apply
        thread = threading.Thread(target=apply)
        thread.start()

    def apply_model_to_image(self, variables):
        # extract from variables
        image_path = variables[0].get()

        # apply
        image = cv2.imread(image_path)
        prediction = self.pca_wavelet_model(np.reshape(image, (1, 64, 64, 3)))

        # navigate
        self.controller.navigate('result')

        # show result image
        self.controller.children['results_screen'].take_info(variables, prediction)

    def configure_process_screen_buttons(self, process_done):
        buttons = self.controller.children['process_screen'].children['content'] \
            .children['process_frame'].children['button_frame'].children['content']
        buttons.children['done_button'].configure(state='enabled' if process_done else 'disabled')
        buttons.children['abort_button'].configure(state='disabled' if process_done else 'enabled')

    def save_model(self):
        # file system dialog
        path = filedialog.asksaveasfilename(
            title="Save a model",
            initialdir=self.initial_dir,
            filetypes=self.filetypes
        )
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
            self.pca_wavelet_model = tf.keras.models.load_model(
                path,
                custom_objects={
                    'Conv2DTransposeSeparableLayer': Conv2DTransposeSeparableLayer,
                    'MeanLayer': MeanLayer,
                    'SymmetricPadding2D': SymmetricPadding2D
                }
            )
            self.controller.navigate('model')

        # start process
        self.controller.navigate('loading')
        thread = threading.Thread(target=load)
        thread.start()

    def give_process_title(self, title):
        self.controller.children['process_screen'].children['title'].configure(text=title)

    def has_data(self):
        return self.pca_wavelet_model is not None

    def get_info(self):
        return [
            ("Name: ",
             [self.name]
             )
        ]
