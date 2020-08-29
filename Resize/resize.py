#!/home/python/anaconda3
import tensorflow as tf
import numpy as np
import matplotlib as plt
from PIL import Image
import os
from pathlib import Path
directory = Path("""/home/python/image/orca""")
global filename
def resize():
    global filename
    temp = "temp"
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            print(filename)
            image = filename
            path = "/home/python/image/resize/" + image
            og = Image.open(image)
            new_image = og.resize((150, 150))
            new_image.save(path)
        else:
            print("failed")

loops = 0
if loops != 638:
    resize()
    loops += 1
else:
    print("Done")
