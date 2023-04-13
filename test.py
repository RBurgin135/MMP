import tensorflow as tf
import numpy as np

from Model.dataset import create_dataset, generate_sem_seg_dataset, generate_infill_dataset
from PCA_Wavelet_Codebase.build import build_model


def build_datasets():
    image_size = (40, 40)
    generate_sem_seg_dataset('C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/semseg')
    generate_infill_dataset('C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill')

    dataset = create_dataset(
        images_path='C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill/images',
        labels_path='C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill/labels'
    )


def test_model():
    dataset = create_dataset('C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill/images',
                             'C:/Users/ritch/Documents/CODING/UNI/Third Year/Major Project/datasets/infill/labels')

    build_model(dataset)






if __name__ == '__main__':
    build_datasets()
