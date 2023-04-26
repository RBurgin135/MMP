import os
import pickle
import threading
import tensorflow as tf

import numpy as np

from PCA_Wavelet_Codebase.custom_layers.conv_2d_transpose_seperable_layer import Conv2DTransposeSeparableLayer
from PCA_Wavelet_Codebase.custom_layers.mean_layer import MeanLayer
from PCA_Wavelet_Codebase.custom_layers.symmetric_padding_2d import SymmetricPadding2D


class Save(threading.Thread):
    def __init__(self, path, controller, image_net, label_net, image_dataset, label_dataset, fully_connected,
                 extra_data):
        super().__init__()
        self.path = path
        self.controller = controller

        # model
        self.image_head = image_net[0]
        self.image_invhead = image_net[1]
        self.label_head = label_net[0]
        self.label_invhead = label_net[1]
        self.A = fully_connected[0]
        self.bias = fully_connected[1]

        # other
        self.extra_data = extra_data
        self.image_dataset = image_dataset
        self.label_dataset = label_dataset

    def run(self):
        # make directory
        os.makedirs(self.path, exist_ok=True)

        # save each to their own file within directory
        # image head
        self.image_head.compile()
        sample = next(iter(self.image_dataset.take(1)))[0]
        sample = tf.reshape(sample, [1, sample.shape[0], sample.shape[1], sample.shape[2]])
        self.image_head._set_inputs(sample)
        self.image_head.save(os.path.join(self.path, 'image_head'))

        # image invhea
        out = self.image_head(sample)
        self.image_invhead(out)
        self.image_invhead._set_inputs(out)
        self.image_invhead.save(os.path.join(self.path, 'image_invhead'))

        # label head
        sample = next(iter(self.label_dataset.take(1)))[0]
        sample = tf.reshape(sample, [1, sample.shape[0], sample.shape[1], sample.shape[2]])
        self.label_head._set_inputs(sample)
        self.label_head.save(os.path.join(self.path, 'label_head'))

        # label invhead
        out = self.label_head(sample)
        self.label_invhead(out)
        self.label_invhead._set_inputs(out)
        self.label_invhead.save(os.path.join(self.path, 'label_invhead'))

        # fully connected
        np.save(os.path.join(self.path, 'A.npy'), self.A)
        np.save(os.path.join(self.path, 'bias.npy'), self.bias)

        # extra data
        pickle.dump(self.extra_data, open(os.path.join(self.path, 'extra_data.pkl'), 'wb'))

        # navigate
        self.controller.navigate('model')


class Load(threading.Thread):
    def __init__(self, controller, model, path):
        super().__init__()
        self.controller = controller
        self.model = model
        self.path = path

    def run(self):
        custom_objects = {'MeanLayer': MeanLayer,
                          'SymmetricPadding2D': SymmetricPadding2D,
                          'Conv2DTransposeSeparableLayer': Conv2DTransposeSeparableLayer}

        try:
            # read from files
            image_head = tf.keras.models.load_model(
                filepath=os.path.join(self.path, 'image_head'),
                custom_objects=custom_objects)
            image_invhead = tf.keras.models.load_model(
                filepath=os.path.join(self.path, 'image_invhead'),
                custom_objects=custom_objects)
            label_head = tf.keras.models.load_model(
                filepath=os.path.join(self.path, 'label_head'),
                custom_objects=custom_objects)
            label_invhead = tf.keras.models.load_model(
                filepath=os.path.join(self.path, 'label_invhead'),
                custom_objects=custom_objects)
            A = np.load(os.path.join(self.path, 'A.npy'))
            bias = np.load(os.path.join(self.path, 'bias.npy'))
            extra_data = pickle.load(open(os.path.join(self.path, 'extra_data.pkl'), 'rb'))

            # commit info
            self.model.image_network = (image_head, image_invhead)
            self.model.label_network = (label_head, label_invhead)
            self.model.fully_connected = (A, bias)
            self.model.take_meta_data(extra_data)
        except IOError:
            print("reset")
            self.model.reset_nets()

        # navigate
        self.controller.navigate('model')
