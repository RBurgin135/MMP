import tensorflow as tf
import numpy as np

from PCA_Wavelet_Codebase.custom_layers.conv_2d_transpose_seperable_layer import Conv2DTransposeSeparableLayer
from PCA_Wavelet_Codebase.custom_layers.mean_layer import MeanLayer
from PCA_Wavelet_Codebase.custom_layers.symmetric_padding_2d import SymmetricPadding2D
from PCA_Wavelet_Codebase.utils import preprocess_dataset, setupfilters3D, filterImg3D, addToPCA, completePCA


def build_model(dataset):
    tf.keras.backend.set_floatx('float64')
    dataset_resize = preprocess_dataset(dataset)

    test_set = dataset_resize.take(500)
    train_set = dataset_resize.skip(500)
    head, inv_head = build1D(
        dataset=train_set,
        count=4,
        samplesize=500,
        keep_percent=1,
        flip=False,
        subtract_mean=True)

    print("model built successfully")

    #built_correctly(head, inv_head, train_set)

    return head, inv_head


def built_correctly(head, invhead, dataset):
    import matplotlib.pyplot as plt

    plt.subplot(221)
    plt.title('Original')
    sample = next(iter(dataset.shuffle(20)))[0]

    plt.imshow(sample*255)
    print("sample.shape", sample.shape)

    img = tf.transpose([sample], [0, 1, 2, 3])
    print("img.shape", img.shape)

    pred = head(img)
    print("pred.shape", pred.shape)

    plt.subplot(222)
    plt.title('Slice')
    plt.imshow(pred[0, :, :, 1] * 5*255, cmap='gray', vmin=0, vmax=1)
    plt.subplot(223)
    plt.title('Slice')
    plt.imshow(pred[0, :, :, 2] * 5*255, cmap='gray', vmin=0, vmax=1)

    print("pred.shape", pred.shape)

    recon = invhead(pred)[0]
    print("recon.shape", recon.shape)
    plt.subplot(224)
    plt.title('Filtered')
    plt.imshow(recon*255)
    plt.show()
    print("sample.dtype", sample.dtype)
    print("recon[0].dtype", recon.dtype)

    print("np.prod(sample.shape)", np.prod(sample.shape))
    psnr = 10 * np.log10(1.0 / ((np.linalg.norm(recon - sample) ** 2) / np.prod(sample.shape)))  # changed
    ncc = np.corrcoef(tf.reshape(sample, [-1]), tf.reshape(recon, [-1]))
    print("psnr = ", psnr)
    print("ncc = ", ncc)


def build1D(dataset, channels=3, count=6, samplesize=-1, keep_percent=0.2, flip=False, activity_regularizer=None,
            inverse_activity_regularizer=None, activation_before=False, subtract_mean=True):
    keep_percent = 4.0 / 9.0 * pow(keep_percent, 1 / float(count))
    print("keep_percent", keep_percent)
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
    print("subset size", len(list(subset)))
    it = iter(subset)
    meanimg = tf.cast(next(it)[0], tf.float64)
    sizex = meanimg.shape[1]
    IMAGE_SIZE_X = sizex
    sizey = meanimg.shape[0]
    IMAGE_SIZE_Y = sizey

    for i in range(1, samplesize):
        meanimg += tf.cast(next(it)[0], tf.float64)
    print("meanimg.dtype", meanimg.dtype)
    meanimg /= float(samplesize)
    if (subtract_mean):
        head.add(MeanLayer(-meanimg))
        invlist.append(MeanLayer(meanimg))

    for lev in range(count):
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
        print("Completing", newsizex)
        pca = pca / float(newsizex * newsizey * samplesize)
        mean /= float(newsizex * newsizey * samplesize)
        print("pca shape", tf.shape(pca))
        s, u = completePCA(pca, mean)
        keep_channels = int(keep_percent * u.shape[1])
        var_explained = 0
        var_total = tf.math.reduce_sum(s, 0)
        s = s / var_total
        var_total_post = tf.math.reduce_sum(s, 0)
        keep_max = channels * (IMAGE_SIZE_Y / filtered.shape[0]) * (IMAGE_SIZE_X / filtered.shape[1])
        print("keep_channels", keep_channels, "keep_max", keep_max)
        compcount = 0
        while var_explained < 1.0 and compcount < keep_max and compcount < keep_channels:
            var_explained += s[compcount]
            compcount += 1

        keep_channels = compcount
        print("keep_channels", keep_channels)
        ufilts = tf.transpose([[[u[:, 0:keep_channels]]]], [0, 1, 2, 3, 4])
        print("ufilts.shape", ufilts.shape)

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
        print("end loop", sizex)

    it = reversed(invlist)
    for e in it:
        invhead.add(e)

    return head, invhead
