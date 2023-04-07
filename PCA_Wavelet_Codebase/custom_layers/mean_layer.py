import tensorflow as tf


class MeanLayer(tf.keras.layers.Layer):
    def __init__(self, mean, **kwargs):
        super(MeanLayer, self).__init__()
        self.mean = tf.keras.backend.variable(tf.keras.backend.cast_to_floatx(mean),
                                              dtype='float64')  # [0:mean.shape[0],0:mean.shape[1],0:mean.shape[2]]
        print("self.mean.dtype", self.mean.dtype)
        super(MeanLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs + self.mean

    def set_mean(self, newmean):
        self.mean = newmean

    def get_mean(self):
        return self.mean

    def get_config(self):
        serial_mean = tf.keras.backend.eval(self.mean)
        print("Getting the config")
        config = super(MeanLayer, self).get_config()
        config.update({'mean': serial_mean})
        print("config = ", config)
        return config

