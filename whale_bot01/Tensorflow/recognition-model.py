#! python
import itertools
import os
import PIL
import PIL.Image
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_hub as hub
import pathlib

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)

dataset = pathlib.Path("Z:/Workspacescurrent/orca-data/")

batch_size = 32
img_height = 150
img_width = 150

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

