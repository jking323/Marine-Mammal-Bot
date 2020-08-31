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
from recognition_model import dataset, img_height, img_width


list_ds = tf.data.Dataset.list_files(str(dataset/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_interations=False)