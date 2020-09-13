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
#from keras.backend.tensorflow_backend import set_session

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)

dataset = pathlib.Path("E:\git\Marine-Mammal-Bot\orca-data")
trainset = pathlib.Path("E:\git\Marine-Mammal-Bot\orca-val")

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

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    trainset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=(batch_size)
)

class_names = train_ds.class_names
print(class_names)

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

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

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

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

#log_dir = "E:\git\Logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = 'E:\git\checkpoints'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callbacks = tf.keras.callbacks.ModelCheckpoint(checkpoint_path=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[cp_callbacks]
#    callbacks=[tensorboard_callback]
)

whale_url = "https://cdn.cnn.com/cnnnext/dam/assets/200728115037-03-orca-whale-pregnant-scn-trnd-exlarge-169.jpg"
whale_path = tf.keras.utils.get_file('whale', origin=whale_url)

img = tf.keras.preprocessing.image.load_img(
    whale_path, target_size=(img_height, img_width)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
