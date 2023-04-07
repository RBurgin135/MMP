import tensorflow as tf

from PCA_Wavelet_Codebase.utils import setupFilts1D, borderMultiplier


class Conv2DTransposeSeparableLayer(tf.keras.layers.Layer):

    def __init__(self, input_shape, **kwargs):
        self.filters, self.invfilters = setupFilts1D()
        # print("input_shape",input_shape)
        self.input_shapeX = [input_shape[0], input_shape[1] + 6, int(input_shape[2])]
        self.tfmultx = borderMultiplier(self.input_shapeX, True)
        self.input_shapeY = [input_shape[0] + 6, input_shape[1] * 2, int(input_shape[2] / 3)]
        self.tfmulty = borderMultiplier(self.input_shapeY, False)

        super(Conv2DTransposeSeparableLayer, self).__init__(**kwargs)

    def get_config(self):
        return {'input_shapeX': self.input_shapeX,
                'tfmultx': self.tfmultx,
                'input_shapeY': self.input_shapeY,
                'tfmulty': self.tfmulty}

    def build(self, input_shape):
        super(Conv2DTransposeSeparableLayer, self).build(input_shape)

    def call(self, inputs):
        output = self.invFilterImgX(inputs, self.tfmultx)
        output = self.invFilterImgY(output, self.tfmulty)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[0] *= 2
        output_shape[1] *= 2
        output_shape[2] = int(output_shape[2] / 9)
        return output_shape

    def invFilterImgX(self, image, tfmulval):
        img = tf.pad(image, [[0, 0], [0, 0], [3, 3], [0, 0]], "SYMMETRIC")
        img = tf.math.multiply(img, tfmulval)
        filter, invfilter = setupFilts1D()

        invfilter = tf.transpose([[invfilter]], [0, 3, 1, 2])
        outputs = []
        for i in range(int(img.shape[3] / 3)):
            slice = tf.gather(img, [3 * i + 0, 3 * i + 1, 3 * i + 2], axis=3)
            im = tf.nn.conv2d_transpose(slice, invfilter, [img.shape[0], img.shape[1], img.shape[2] * 2 + 7, 1], [1, 2],
                                        padding='VALID', data_format='NHWC')
            outputs.append(im)

        outimg = tf.stack(outputs)
        outimg = tf.transpose(outimg, perm=[1, 2, 3, 0, 4])[:, :, :, :, 0]

        return outimg[:, :, 10:outimg.shape[2] - 9, :]

    def invFilterImgY(self, image, tfmulval):
        img = tf.pad(image, [[0, 0], [3, 3], [0, 0], [0, 0]], "SYMMETRIC")
        img = tf.math.multiply(img, tfmulval)
        filter, invfilter = setupFilts1D()

        invfilter = tf.transpose([[invfilter]], [3, 0, 1, 2])
        outputs = []
        for i in range(int(img.shape[3] / 3)):
            slice = tf.gather(img, [3 * i + 0, 3 * i + 1, 3 * i + 2], axis=3)
            im = tf.nn.conv2d_transpose(slice, invfilter, [img.shape[0], img.shape[1] * 2 + 7, img.shape[2], 1], [2, 1],
                                        padding='VALID', data_format='NHWC')
            outputs.append(im)

        outimg = tf.stack(outputs)
        outimg = tf.transpose(outimg, perm=[1, 2, 3, 0, 4])[:, :, :, :, 0]
        outimg = outimg[:, 10:outimg.shape[1] - 9, :, :]

        return outimg
