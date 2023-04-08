import tensorflow as tf


class SymmetricPadding2D(tf.keras.layers.Layer):

    def __init__(self, output_dim, padding=[1, 1],
                 data_format="channels_last", **kwargs):
        self.output_dim = output_dim
        self.data_format = data_format
        self.padding = padding
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def get_config(self):
        return {'output_dim': self.output_dim,
                'data_format': self.data_format,
                'padding': self.padding}

    def build(self, input_shape):
        super(SymmetricPadding2D, self).build(input_shape)

    def call(self, inputs):
        if self.data_format == "channels_last":
            pad = [[0, 0]] + [[i, i] for i in self.padding] + [[0, 0]]
        elif self.data_format == "channels_first":
            pad = [[0, 0], [0, 0]] + [[i, i] for i in self.padding]

        paddings = tf.constant(pad)
        out = tf.pad(inputs, paddings, "SYMMETRIC")
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim
