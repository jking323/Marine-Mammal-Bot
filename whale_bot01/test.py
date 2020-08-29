import tensorflow as tf
import numpy as np
import matplotlibas plt
import pillow
import os

directory = 

for filename in os.listdir(directory):
	if filename.endswith(".jpg"):
		image = filename
	else:
		continue

def resize(image):
	Image.open(image)
	new_image = image.resize((150, 150))
	new_image.save(filename)
	