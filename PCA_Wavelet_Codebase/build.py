import tensorflow as tf
import numpy as np

from GUI.screens.console import Console
from PCA_Wavelet_Codebase.custom_layers.conv_2d_transpose_seperable_layer import Conv2DTransposeSeparableLayer
from PCA_Wavelet_Codebase.custom_layers.mean_layer import MeanLayer
from PCA_Wavelet_Codebase.custom_layers.symmetric_padding_2d import SymmetricPadding2D
from PCA_Wavelet_Codebase.utils import setupfilters3D, filterImg3D, addToPCA, completePCA


def map_fully_connected(image, image_network, inverse_label_network, A, bias):
    decomp = image_network(image)
    shape = decomp.shape
    decomp = tf.reshape(decomp, [-1])

    decomp = tf.linalg.matvec(A, decomp, transpose_a=True)
    decomp = tf.math.add(decomp, bias)
    pred = tf.reshape(decomp, [shape[0], shape[1], shape[2], -1])

    recon = inverse_label_network(pred)
    return recon


def build_fully_connected(image_network, label_network, image_set, label_set):
    it = iter(label_set)
    firstit = True
    total = len(image_set)
    count = 0

    for img in image_set:
        pimg = next(it)
        decomp = image_network(img)
        pdecomp = label_network(pimg)
        if (firstit):
            firstit = False
            channels = decomp.shape[1] * decomp.shape[2] * decomp.shape[3]
            pchannels = pdecomp.shape[1] * pdecomp.shape[2] * pdecomp.shape[3]
            totalcount = 0
            xxt = tf.zeros([channels, channels], dtype=tf.float64)
            yxt = tf.zeros([channels, pchannels], dtype=tf.float64)
            y = tf.zeros([pchannels], dtype=tf.float64)
            x = tf.zeros([channels], dtype=tf.float64)

        totalcount += 1
        mat = tf.reshape(decomp, [-1])
        pmat = tf.reshape(pdecomp, [-1])
        cov = tf.linalg.matmul([mat], [mat], transpose_a=True)
        pcov = tf.linalg.matmul([mat], [pmat], transpose_a=True)
        xxt = xxt + cov
        yxt = yxt + pcov
        x = x + mat
        y = y + pmat

        count += 1
        Console.print(f"training: {round((count/total)*100)}%")

    Console.print("training complete")
    Console.print("calculating xxt")
    xxt = xxt - tf.linalg.matmul([x], [x], transpose_a=True) / totalcount
    A = np.linalg.pinv(xxt)
    Console.print("calculating yxt")
    yxt = yxt - tf.linalg.matmul([x], [y], transpose_a=True) / totalcount
    A = A @ yxt
    Console.print("calculating bias")
    bias = (y - tf.linalg.matvec(A, x, transpose_a=True)) / totalcount
    Console.print("fully connected built")
    return A, bias


def build_1d(dataset, channels=3, layers=6, samplesize=-1, keep_percent=0.2, flip=False, activity_regularizer=None,
             inverse_activity_regularizer=None, activation_before=False, subtract_mean=True):
    keep_percent = 4.0 / 9.0 * pow(keep_percent, 1 / float(layers))
    Console.print(f"keep_percent {keep_percent}")
    head = tf.keras.Sequential()
    head.run_eagerly = True
    invhead = tf.keras.Sequential()
    invlist = []
    invbinit = tf.constant_initializer(np.zeros(3))
    if samplesize < 0:
        subset = dataset
        samplesize = len(list(subset))
    else:
        subset = dataset.take(samplesize)

    if flip:
        flipped = subset.map(lambda x, y: reverse(x, y))
        subset = subset.concatenate(flipped)
        samplesize *= 2
    Console.print(f"subset size {len(list(subset))}")
    it = iter(subset)
    meanimg = tf.cast(next(it)[0], tf.float64)
    sizex = meanimg.shape[1]
    IMAGE_SIZE_X = sizex
    sizey = meanimg.shape[0]
    IMAGE_SIZE_Y = sizey

    for i in range(1, samplesize):
        meanimg += tf.cast(next(it)[0], tf.float64)
    Console.print(f"meanimg.dtype {meanimg.dtype}")
    meanimg /= float(samplesize)
    if (subtract_mean):
        head.add(MeanLayer(-meanimg))
        invlist.append(MeanLayer(meanimg))

    for lev in range(layers):
        outchan = channels * 9
        pca = tf.zeros([outchan, outchan], dtype=tf.float64)
        mean = tf.zeros(outchan, dtype=tf.float64)
        filts3D = setupfilters3D(channels)
        filts3D = tf.transpose(filts3D, [2, 3, 1, 0])
        newsizex = sizex / 2
        newsizey = sizey / 2
        for image in subset:
            img = tf.cast(image[0], tf.float64)
            img = tf.transpose([img], [0, 1, 2, 3])

            pred = head(img)[0]
            filtered = filterImg3D(pred, filts=filts3D)
            pca, mean = addToPCA(filtered, pca, mean)
        Console.print(f"Completing {newsizex}")
        pca = pca / float(newsizex * newsizey * samplesize)
        mean /= float(newsizex * newsizey * samplesize)
        Console.print(f"pca shape {tf.shape(pca)}")
        s, u = completePCA(pca, mean)
        keep_channels = int(keep_percent * u.shape[1])
        var_explained = 0
        var_total = tf.math.reduce_sum(s, 0)
        s = s / var_total
        var_total_post = tf.math.reduce_sum(s, 0)
        keep_max = channels * (IMAGE_SIZE_Y / filtered.shape[0]) * (IMAGE_SIZE_X / filtered.shape[1])
        Console.print(f"keep_channels {keep_channels}, keep_max {keep_max}")
        compcount = 0
        while var_explained < 1.0 and compcount < keep_max and compcount < keep_channels:
            var_explained += s[compcount]
            compcount += 1

        keep_channels = compcount
        Console.print(f"keep_channels {keep_channels}")
        ufilts = tf.transpose([[[u[:, 0:keep_channels]]]], [0, 1, 2, 3, 4])
        Console.print(f"ufilts.shape {ufilts.shape}")

        filts3D = tf.transpose([filts3D], [0, 3, 1, 2, 4])
        newfilt = tf.nn.conv3d(filts3D, ufilts, [1, 1, 1, 1, 1], 'VALID', data_format='NDHWC')
        filtsOrig = tf.transpose(newfilt[0], [1, 2, 0, 3])
        numpynewfilt = filtsOrig.numpy()
        init = tf.constant_initializer(numpynewfilt)
        bias = -tf.linalg.matvec(ufilts, mean, transpose_a=True)[0, 0, 0]
        binit = tf.constant_initializer(bias.numpy())
        if (activation_before):
            head.add(tf.keras.layers.Activation(activity_regularizer))

        head.add(SymmetricPadding2D(2, input_shape=(int(sizey), int(sizex), channels), padding=[2, 2]))
        head.add(tf.keras.layers.Conv2D(keep_channels, (5, 5), strides=(2, 2),
                                        input_shape=(int(sizey) + 4, int(sizex) + 4, channels),
                                        kernel_initializer=init, bias_initializer=binit))
        if (not activation_before):
            head.add(tf.keras.layers.Activation(activity_regularizer))

        target_shape = [filtered.shape[0], filtered.shape[1], u.shape[1]]
        if (activation_before):
            invlist.append(tf.keras.layers.Activation(inverse_activity_regularizer))

        invlist.append(Conv2DTransposeSeparableLayer(target_shape))
        utfilts = tf.transpose([[u[:, 0:keep_channels]]], [0, 1, 3, 2])
        binit = tf.constant_initializer(mean.numpy())
        kinit = tf.constant_initializer(utfilts.numpy())

        invlist.append(tf.keras.layers.Conv2D(u.shape[0], [1, 1], strides=1, padding='VALID', use_bias=True,
                                              kernel_initializer=kinit, bias_initializer=binit))
        if not activation_before:
            invlist.append(tf.keras.layers.Activation(inverse_activity_regularizer))

        channels = keep_channels
        sizex = newsizex
        sizey = newsizey
        Console.print(f"end loop {sizex}")

    it = reversed(invlist)
    for e in it:
        invhead.add(e)

    return head, invhead
