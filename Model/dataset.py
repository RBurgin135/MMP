import numpy as np
import random
import cv2
import tensorflow as tf
import os

# set parameters
image_size = (64, 64)
batch_size = 32
min_radius = 5
max_radius = 10


def create_dataset(images_path, labels_path):

    # define generator
    def generator():
        for filename in os.listdir(images_path):
            # load image
            img = cv2.imread(f'{images_path}/{filename}')
            if img is None:
                print(f"Error loading image, with path: {images_path}/{filename}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.

            # load label
            label = cv2.imread(f'{labels_path}/{filename}', cv2.IMREAD_GRAYSCALE)
            if label is None:
                print(f"No equivalent label file, with path: {labels_path}/{filename}")
            label = label.astype(np.float32) / 255.
            label = np.expand_dims(label, axis=-1)

            # create dictionary
            yield {'image': img, 'label': label}

    # create dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types={'image': tf.float32, 'label': tf.float32}
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def generate_infill_dataset(dataset_path):
    root_path = dataset_path
    image_path = f'{root_path}/images'
    label_path = f'{root_path}/labels'

    num_images = 1000
    # generate images and labels
    for i in range(num_images):
        # create a new image
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

        # randomly place one circle with a random color
        center = (
            random.randint(max_radius, image_size[0] - max_radius),
            random.randint(max_radius, image_size[1] - max_radius))
        radius = random.randint(min_radius, max_radius)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(image, center, radius, color, -1)

        # randomly place a square inside the circle
        square_size = radius // 2
        square_center = (
            random.randint(center[0] - radius + square_size, center[0] + radius - square_size),
            random.randint(center[1] - radius + square_size, center[1] + radius - square_size)
        )
        cv2.rectangle(image,
                      (square_center[0] - square_size, square_center[1] - square_size),
                      (square_center[0] + square_size, square_center[1] + square_size),
                      (0, 0, 0),
                      -1)

        # save image as .png
        cv2.imwrite(f'{image_path}/{i}.png', image)

        # create label as filled circle
        label = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        cv2.circle(label, center, radius, color, -1)

        # save label as .png
        cv2.imwrite(f'{label_path}/{i}.png', label)


def generate_sem_seg_dataset(dataset_path):
    root_path = dataset_path
    image_path = f'{root_path}/images'
    label_path = f'{root_path}/labels'

    num_images = 1000
    # generate images and labels
    for i in range(num_images):
        # create a new image
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

        # randomly place one circle with a random color
        center = (
            random.randint(max_radius, image_size[0] - max_radius),
            random.randint(max_radius, image_size[1] - max_radius))
        radius = random.randint(min_radius, max_radius)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(image, center, radius, color, -1)

        # save image as .png
        cv2.imwrite(f'{image_path}/{i}.png', image)

        # create binary mask
        mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        # convert mask to binary
        mask = mask / 255

        # save binary mask as .png
        cv2.imwrite(f'{label_path}/{i}.png', mask)
