#! python
import matplotlib.pyplot as plt
import itertools
import os
import PIL
import PIL.Image
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import pathlib
#from recognition_model import dataset

directory = pathlib.Path('E:\git\Marine-Mammal-Bot\orca-data')
#
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(150, 150),
        shuffle=True,
        seed=None,
        interpolation="bilinear",
        follow_links=False,)

image_inputs = tf.keras.Input(shape=(150, 150, 3))
dense = layers.Dense(64, activation="relu")
x = dense(image_inputs)
x = layers.Conv2D(filters=32, kernel_size=(3 ,3), activation="relu")
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
model = tf.keras.Model(inputs=image_inputs, outputs=outputs, name="mnist_model")
model.summary()
"""
tf.keras.utils.plot_model(model, "my_first_model.png")
"""
