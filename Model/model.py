# from PCA_wavelet_codebase.admm import build_model


class Model:
    def __init__(self, controller):
        self.has_data = False
        self.controller = controller
        self.pca_wavelet_model = None
        self.name = None
        self.input_set = None
        self.label_set = None

    def create_new_model(self, variables):
        print("create new model")
        self.has_data = True
        self.name = variables[0]
        self.input_set = variables[1]
        self.label_set = variables[2]
        # self.pca_wavelet_model, _, _, _ = build_model()

    def new_input_set(self, path):
        print("new input set: "+path)

    def new_label_set(self, path):
        print("new label set: "+path)
