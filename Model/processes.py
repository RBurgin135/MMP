import os
import threading

import cv2
import numpy as np

from Model.dataset import create_dataset
from PCA_Wavelet_Codebase.build import build_model


class Process(threading.Thread):
    def __init__(self, current_model):
        super().__init__()
        self.model = current_model
        self.stop_flag = threading.Event()

    def abort(self):
        self.stop_flag.set()

    def configure_process_screen_buttons(self, process_done):
        buttons = self.model.controller.children['process_screen'].children['content'] \
            .children['process_frame'].children['button_frame'].children['content']
        buttons.children['done_button'].configure(state='enabled' if process_done else 'disabled')
        buttons.children['abort_button'].configure(state='disabled' if process_done else 'enabled')

    def abort_check(self):
        return self.stop_flag.is_set()


class Build(Process):
    def __init__(self, current_model, images_path, labels_path):
        super().__init__(current_model)
        self.images_path = images_path
        self.labels_path = labels_path

    def run(self):
        # create dataset
        dataset = create_dataset(self.images_path, self.labels_path)
        if self.abort_check(): return

        # build
        head, invhead = build_model(dataset)
        if self.abort_check(): return
        self.model.pca_wavelet_model = head

        # configure buttons
        self.configure_process_screen_buttons(process_done=True)


class ApplyToDir(Process):
    def __init__(self, current_model, images_path, output_path):
        super().__init__(current_model)
        self.images_path = images_path
        self.output_path = output_path

    def run(self):
        # iterate over images
        for x in os.listdir(self.images_path):
            # abort check
            if self.abort_check(): return

            # apply
            image = cv2.imread(self.images_path + x)
            prediction = self.model.pca_wavelet_model(
                np.reshape(image, (1, 64, 64, 3))
            )

            # save
            cv2.imwrite(self.output_path + x, np.array(prediction[0, :, :, 1] * 255))
            print(f"finished: {x}")

        # configure buttons
        self.configure_process_screen_buttons(process_done=True)
