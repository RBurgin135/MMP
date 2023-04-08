import tensorflow as tf
import numpy as np

IMAGE_SIZE = 64


def preprocess_dataset(dataset):
    processed_data = [[pre_process_image(x['image'])] for x in dataset]
    processed_dataset = tf.data.Dataset.from_tensor_slices(processed_data)

    return processed_dataset


def pre_process_image(image):
    print("pre_process image.shape", image.shape)
    image = tf.cast(image, tf.float64)
    image = image / 255.0
    print("pre_process image.shape resized", image.shape, image.dtype)
    # image = tf.image.central_crop(image,0.5)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # resize changes type to float32!

    print("pre_process image.shape resized", image.shape, image.dtype)
    image = tf.cast(image, tf.float64)
    print("pre_process image.shape resized", image.shape, image.dtype)

    return image


def pre_process_entry(image, label):
    image = pre_process_image(image)
    return image, label


def addToPCA(ten, pca, mean):
    mat = tf.reshape(ten, [-1, ten.shape[2]])
    cov = tf.tensordot(mat, mat, [0, 0])
    m = tf.ones(mat.shape[0], dtype=tf.float64)
    m = tf.linalg.matvec(mat, m, transpose_a=True)
    pca = pca + cov
    mean = mean + m

    return pca, mean


def completePCA(pca, mean):
    mouter = tf.tensordot(mean, mean, axes=0)
    pca -= mouter
    s, u, v = tf.linalg.svd(pca)
    return s, u  # ,v


def setupfilters3D(channels):
    filts = tf.constant(
        [[1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0], [0.0, -1.0 / 4.0, 2.0 / 4.0, -1.0 / 4.0, 0.0],
         [0.0, 1.0 / 2.0, 0.0, -1.0 / 2.0, 0.0]], dtype=tf.float64)
    filts3D = []
    for k in range(channels):
        for i in range(3):
            for j in range(3):
                filt2D = tf.pad([tf.tensordot(filts[i], filts[j], axes=0)], [[k, channels - k - 1], [0, 0], [0, 0]],
                                mode="CONSTANT", constant_values=0)
                filts3D.append(filt2D)
    filters = tf.stack(filts3D)

    return filters


def filterImg3D(image, filts=None):
    if filts is None:
        filts = setupfilters3D(image.shape[2])
        filts = tf.transpose(filts, [2, 3, 1, 0])
    img = tf.pad([image], [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")
    img = tf.nn.conv2d(img, filts, [1, 2, 2, 1], 'VALID', data_format='NHWC')
    return img[0]


def setupFilts1D():
    filts = tf.constant(
        [[1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0], [0.0, -1.0 / 4.0, 2.0 / 4.0, -1.0 / 4.0, 0.0],
         [0.0, 1.0 / 2.0, 0.0, -1.0 / 2.0, 0.0]], dtype=tf.float64)
    smooth = [0.0, 0.0, 1.0 / 16.0, 0.5, 14.0 / 16.0, 0.5, 1.0 / 16.0, 0.0, 0.0]
    even = [-1.0 / 128.0, -1.0 / 16.0, -10.0 / 64.0, -7.0 / 16.0, 85.0 / 64.0, -7.0 / 16.0, -10.0 / 64.0, -1.0 / 16.0,
            -1.0 / 128.0]
    odd = [1.0 / 256.0, 1.0 / 32.0, 15.0 / 128.0, 17.0 / 32.0, 0.0, -17.0 / 32.0, -15.0 / 128.0, -1.0 / 32.0,
           -1.0 / 256.0]
    invfilts = tf.constant([smooth, even, odd], dtype=tf.float64)
    return filts, invfilts


def borderMultiplier(shape, xAxis):
    mulval = np.ones(shape)
    if (xAxis):
        for i in range(3):
            for j in range(shape[0]):
                for k in range(int(shape[2] / 3)):
                    mulval[j, i, 2 + 3 * k] = -1
                    mulval[j, shape[1] - i - 1, 2 + 3 * k] = -1
    else:
        for i in range(3):
            for j in range(shape[1]):
                for k in range(int(shape[2] / 3)):
                    mulval[i, j, 2 + 3 * k] = -1
                    mulval[shape[0] - i - 1, j, 2 + 3 * k] = -1

    return mulval
