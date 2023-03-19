class Model:
    def __init__(self, controller):
        self.name = None
        self.controller = controller
        self.pca_wavelet_model = None
        self.input_set = None
        self.label_set = None

    def create_new_model(self, variables):
        self.name = variables[0].get()
        self.input_set = variables[1].get()
        self.label_set = variables[2].get()
        self.controller.navigate("model")

    def has_data(self):
        return self.name is not None

    def get_info(self):
        return [
            ("Name: ",
             [self.name]
             ),
            ("Datasets:",
             ["Input set: " + self.input_set,
              "Label set: " + self.label_set]
             )
        ]
