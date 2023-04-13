import threading
from tkinter import filedialog
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
            ('All files', '*.*')
        )
        self.initial_dir = ""

    def create_new_model(self, variables):
        # extract from variables
        self.name = variables[0].get()
        images_path = variables[1].get()
        labels_path = variables[2].get()

        # navigate
        self.controller.navigate("process")

        # train model subroutine
        def train_model():
            dataset = create_dataset(images_path, labels_path)
            self.pca_wavelet_model, _ = build_model(dataset)

            # enable done button
            buttons = self.controller.children['process_screen'].children['content']\
                .children['process_frame'].children['button_frame'].children['content']
            buttons.children['done_button'].configure(state='enabled')
            buttons.children['abort_button'].configure(state='disabled')

        # build
        thread = threading.Thread(target=train_model)
        thread.start()

    def apply_model(self, variables):
        # extract from variables
        images_path = variables[1].get()
        labels_path = variables[2].get()

        # apply
        dataset = create_dataset(images_path, labels_path)
        self.pca_wavelet_model(dataset)

    def save(self):
        path = filedialog.asksaveasfilename(
            title="Save a model",
            initialdir=self.initial_dir,
            filetypes=self.filetypes
        )
        self.pca_wavelet_model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.pca_wavelet_model.save(path+".h5")

    def load(self):
        path = filedialog.askopenfilename(
            title="Load a model",
            initialdir=self.initial_dir,
            filetypes=self.filetypes
        )
        self.pca_wavelet_model = tf.keras.models.load_model(
            path,
            custom_objects={
                'Conv2DTransposeSeparableLayer': Conv2DTransposeSeparableLayer,
                'MeanLayer': MeanLayer,
                'SymmetricPadding2D': SymmetricPadding2D
            }
        )

    def has_data(self):
        return self.name is not None

    def get_info(self):
        return [
            ("Name: ",
             [self.name]
             )
        ]
