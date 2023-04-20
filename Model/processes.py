import os
import threading
from tkinter import messagebox

import cv2
import numpy as np
import tensorflow as tf

from Model.dataset import create_dataset
from PCA_Wavelet_Codebase.build import build_1d, build_fully_connected, map_fully_connected
from PCA_Wavelet_Codebase.utils import preprocess_dataset, pre_process_image


class Process(threading.Thread):
    def __init__(self, current_model, controller):
        super().__init__()
        self.controller = controller
        self.model = current_model
        self.stop_flag = threading.Event()

    def abort(self, notify=False):
        self.stop_flag.set()
        self.controller.navigate("model")
        if notify:
            messagebox.showerror(
                title='File Not Found',
                message='Error caused by reading files from system. Check all files and directories input are correct.'
            )

    def configure_process_screen_buttons(self, process_done):
        buttons = self.model.controller.children['process_screen'].children['process_frame']\
            .children['button_frame'].children['content']
        buttons.children['done_button'].configure(state='enabled' if process_done else 'disabled')
        buttons.children['abort_button'].configure(state='disabled' if process_done else 'enabled')

    def abort_check(self):
        return self.stop_flag.is_set()


class Build(Process):
    def __init__(self, current_model, images_path, labels_path, count, layers, controller):
        super().__init__(current_model, controller)
        self.images_path = images_path
        self.labels_path = labels_path
        self.count = count
        self.layers = layers
        self.dataset = None

    def run(self):
        # create dataset
        self.dataset = create_dataset(self.images_path, self.labels_path)

        if self.abort_check(): return
        print("dataset created")

        # build
        try:
            i_n, l_n, f_c = self.build_model()
            if self.abort_check(): return
            self.model.image_network = i_n
            self.model.label_network = l_n
            self.model.fully_connected = f_c
        except Exception:
            self.abort(notify=True)
        print("model built successfully")

        # configure buttons
        self.configure_process_screen_buttons(process_done=True)

    def build_model(self):
        tf.keras.backend.set_floatx('float64')
        image_set = preprocess_dataset(self.dataset, 'image')
        label_set = preprocess_dataset(self.dataset, 'label')
        if self.abort_check(): return None, None, None

        # build image autoencoder
        image_head, image_inv_head = build_1d(
            dataset=image_set.take(self.count),
            layers=self.layers,
            samplesize=self.count,
            keep_percent=1,
            flip=False,
            subtract_mean=True)
        if self.abort_check(): return None, None, None
        print("built image autoencoder")

        # build label autoencoder
        label_head, label_inv_head = build_1d(
            dataset=label_set.take(self.count),
            layers=self.layers,
            samplesize=self.count,
            keep_percent=1,
            flip=False,
            subtract_mean=True
        )
        if self.abort_check(): return None, None, None
        print("built label autoencoder")

        # build fully connected
        A, bias = build_fully_connected(
            image_network=image_head,
            label_network=label_head,
            image_set=image_set.take(self.count),
            label_set=label_set.take(self.count)
        )
        if self.abort_check(): return None, None, None
        print("built fully connected network")

        return (image_head, image_inv_head), (label_head, label_inv_head), (A, bias)


class ApplyToDir(Process):
    def __init__(self, current_model, images_path, output_path, controller):
        super().__init__(current_model, controller)
        self.images_path = images_path
        self.output_path = output_path

    def run(self):
        # iterate over images
        for x in os.listdir(self.images_path):
            # abort check
            if self.abort_check(): return

            # apply
            try:
                image = cv2.imread(self.images_path + x)
                image = image[np.newaxis, ...]
                image = pre_process_image(image)

                reconstructed = map_fully_connected(
                    image=image,
                    image_network=self.model.image_network[0],
                    inverse_label_network=self.model.label_network[1],
                    A=self.model.fully_connected[0],
                    bias=self.model.fully_connected[1]
                )

                # save
                reconstructed = np.uint8(np.squeeze(np.array(reconstructed)))
                cv2.imwrite(self.output_path + x, reconstructed)
                print(f"finished: {x}")
            except Exception:
                self.abort(notify=True)

        # configure buttons
        self.configure_process_screen_buttons(process_done=True)
