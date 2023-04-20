import threading
from tkinter import filedialog

import cv2
import numpy as np
from PIL import ImageTk, Image

from Model import processes
from PCA_Wavelet_Codebase.build import map_fully_connected
from PCA_Wavelet_Codebase.utils import pre_process_image


class Model:
    def __init__(self, controller):
        # info
        self.name = None
        self.count = None
        self.layers = None
        self.controller = controller

        # networks
        self.image_network = None
        self.label_network = None
        self.fully_connected = None

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
        self.count = int(variables[3].get())
        self.layers = int(variables[4].get())

        # build
        thread = processes.Build(
            current_model=self,
            images_path=images_path,
            labels_path=labels_path,
            count=self.count,
            layers=self.layers,
            controller=self.controller
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
            output_path=output_path,
            controller=self.controller
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
        image = image[np.newaxis, ...]
        image = pre_process_image(image)

        reconstructed = map_fully_connected(
            image=image,
            image_network=self.image_network[0],
            inverse_label_network=self.label_network[1],
            A=self.fully_connected[0],
            bias=self.fully_connected[1]
        )

        # navigate
        self.controller.navigate('result')

        # show result image
        self.give_results_prediction(reconstructed)

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
            # TODO self.pca_wavelet_model.compile(optimizer='adam', loss='categorical_crossentropy')
            #self.pca_wavelet_model.save(path)
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
            #try:
            #    self.pca_wavelet_model = tf.keras.models.load_model(
            #        path,
            #        custom_objects={
            #            'Conv2DTransposeSeparableLayer': Conv2DTransposeSeparableLayer,
            #            'MeanLayer': MeanLayer,
            #            'SymmetricPadding2D': SymmetricPadding2D
            #        }
            #    )
            #except IOError:
            #    pass
            self.controller.navigate('model')

        # start process
        self.controller.navigate('loading')
        thread = threading.Thread(target=load)
        thread.start()

    def give_process_info(self, title, process):
        # set title
        process_frame = self.controller.children['process_screen'].children['process_frame']
        process_frame.children['title'].configure(text=title)

        # set process
        button_frame = process_frame.children['button_frame']
        button_frame.process = process

    def give_results_prediction(self, prediction):
        # get references
        results_screen = self.controller.children['results_screen']
        button_frame = results_screen.children['button_frame']

        # convert types
        prediction = np.uint8(np.squeeze(np.array(prediction)))

        # take images
        button_frame.cv2_image = prediction
        image = Image.fromarray(prediction)
        results_screen.show_image = ImageTk.PhotoImage(
            image=image.resize((200, 200), Image.NEAREST)
        )

        # reconfigure result image
        results_screen.children['result_image'].configure(image=results_screen.show_image)

    def has_data(self):
        return self.image_network is not None

    def get_info(self):
        return [
            ("Name: ",
             [self.name]
             )
        ]
