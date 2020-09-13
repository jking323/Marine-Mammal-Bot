import tensorflow as tf
import numpy as np 

img_height = 150
img_width = 150

dataset = pathlib.Path("E:\git\Marine-Mammal-Bot\orca-data")
trainset = pathlib.Path("E:\git\Marine-Mammal-Bot\orca-val")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    trainset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=(batch_size)
)

saved_model = tf.keras.models.load_model()