import threading

from Model.dataset import create_dataset
from PCA_Wavelet_Codebase.build import build_model


class Model:
    def __init__(self, controller):
        self.name = None
        self.controller = controller
        self.pca_wavelet_model = None

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

    def has_data(self):
        return self.name is not None

    def get_info(self):
        return [
            ("Name: ",
             [self.name]
             )
        ]
