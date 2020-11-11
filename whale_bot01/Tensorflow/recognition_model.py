#import all of the dependencies for the program
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
import datetime


#prints the current version of tensorflow

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)

#defines data and training set directories

dataset = pathlib.Path("C:\whale\orca-data")
trainset = pathlib.Path("C:\whale\orca-val")

#defines processing batch size and image dimensions

batch_size = 32
img_height = 150
img_width = 150

#defines training preprocessing pipeline

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

#defines validation image pipeline

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    trainset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=(batch_size)
)
#prints class names to output

class_names = train_ds.class_names
print(class_names)

#prints image information to output

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


#Normalizes data values I believe this sets values to a decimel number between 0 and 1

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.experimental.AUTOTUNE

#caches datasets in memory to prevent I/O bottlenecks

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

#defines the model

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

#compiles the model and allows it to be save for furture use

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

#log_dir = "E:\git\Logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#defines training checkpoint paths these checkpoints allow for me to call the model at different points in the 
#training process and ensures that if an error occurs during one of the training rounds all of the data isn't lost

checkpoint_path = 'E:\git\checkpoints\cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
#defines the checkpoint function
cp_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
#trains the moedl, typically the callbacks would be uncommented to allow checkpints
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
#    callbacks=[cp_callbacks]
#    callbacks=[tensorboard_callback]
)
#Downloads test image to check the model

whale_url = "https://images-na.ssl-images-amazon.com/images/I/71h68eptZwL._AC_SX466_.jpg"
whale_path = tf.keras.utils.get_file('whale', origin=whale_url)
#processes the test image into a format that the model can understand
img = tf.keras.preprocessing.image.load_img(
    whale_path, target_size=(img_height, img_width)
)
#converts test image to array

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

#uses model to make prediction

predictions = model.predict(img_array)

#quatifies prediction confidence as a %

score = tf.nn.softmax(predictions[0])

#prints prediction and percentage as human readable output

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

