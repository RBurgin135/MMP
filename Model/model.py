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

        # build
        dataset = create_dataset(images_path, labels_path)
        self.pca_wavelet_model, _ = build_model(dataset)

        self.controller.navigate("model")

    def has_data(self):
        return self.name is not None

    def get_info(self):
        return [
            ("Name: ",
             [self.name]
             )
        ]
