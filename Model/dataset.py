import numpy as np
import random
import cv2
import tensorflow as tf
import os

# set parameters
image_size = (40, 40)
batch_size = 32
min_radius = 5
max_radius = 10


def create_dataset(dataset_path):
    # define paths
    root_path = dataset_path
    image_path = f'{root_path}/images'
    label_path = f'{root_path}/labels'

    # define generator
    def generator():
        for filename in os.listdir(image_path):
            # load image
            img = cv2.imread(f'{image_path}/{filename}')
            if img is None:
                print(f"Error loading image, with path: {image_path}/{filename}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.

            # load label
            label = cv2.imread(f'{label_path}/{filename}', cv2.IMREAD_GRAYSCALE)
            if label is None:
                print(f"No equivalent label file, with path: {label_path}/{filename}")
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


def generate_images(dataset_path):
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
