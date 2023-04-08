import tensorflow as tf
import numpy as np

from Model.dataset import create_dataset, generate_sem_seg_dataset, generate_infill_dataset
from PCA_Wavelet_Codebase.build import build_model


def test_dataset():
    image_size = (40, 40)
    generate_sem_seg_dataset('C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/semseg')
    generate_infill_dataset('C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill')

    dataset = create_dataset('C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill')

    # define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, 1, activation='sigmoid')
    ])

    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(dataset, epochs=10)


def test_model():
    dataset = create_dataset('C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill')

    build_model(dataset)






if __name__ == '__main__':
    test_model()
