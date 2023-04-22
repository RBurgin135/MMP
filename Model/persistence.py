import os
import pickle
import threading

import numpy as np


class Save(threading.Thread):
    def __init__(self, path, controller, image_net, label_net, fully_connected, extra_data):
        super().__init__()
        self.path = path
        self.controller = controller
        self.image_net = image_net
        self.label_net = label_net
        self.fully_connected = fully_connected
        self.extra_data = extra_data

    def run(self):
        # make directory
        os.makedirs(self.path, exist_ok=True)

        # save each to their own file within directory
        for i in range(2):
            # image net
            if i == 0:
                self.image_net[i].compile(optimizer='adam', loss='categorical_crossentropy')
            else:
                self.image_net[i].build(self.image_net[0].input_shape)
            self.image_net[i].save(os.path.join(self.path, f'pwn_image_net_{i}.h5'))

            # label net
            if i == 0:
                self.label_net[i].compile(optimizer='adam', loss='categorical_crossentropy')
            else:
                self.image_net[i].build(self.image_net[0].input_shape)
            self.label_net[i].save(os.path.join(self.path, f'pwn_label_net_{i}.h5'))

            # fully connected
            np.save(os.path.join(self.path, f'pwn_fully_connected_{i}.npy'), self.fully_connected[i].numpy())

        # extra data
        pickle.dump(self.extra_data, open(os.path.join(self.path, f'extra_data.pkl'), 'wb'))

        # navigate
        self.controller.navigate('model')


class Load (threading.Thread):
    def __init__(self, controller, path):
        super().__init__()
        self.controller = controller
        self.path = path

    def run(self):
        try:
            pass  # TODO
        except IOError:
            pass

        # navigate
        self.controller.navigate('model')
