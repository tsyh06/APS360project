import inkml2img
import pandas as pd
import os
import glob
import io
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
from PIL import ImageDraw
from pathlib import Path

def convert_inkml_img():
    directory = Path(os.getcwd() + "/all_inkml/crohm 2013")
    inkml_files = directory.glob('*.inkml')

    # Convert inkml to png
    for file in inkml_files:
        strFile = str(file)[:-6]
        print(strFile)
        inkml2img.inkml2img(file, strFile + ".png")


def expand2square(image, background_color):
    width, height = image.size   # Get dimensions

    if width == height:
        return image
    if width > height:
        result = Image.new(image.mode, (width, width), background_color)
        result.paste(image, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(image.mode, (height, height), background_color)
        result.paste(image, ((height - width) // 2, 0))
        return result


def imgs2squares():
    directory = Path(os.getcwd() + "/all_inkml/crohm 2013")
    print(directory)
    png_files = directory.glob('*.PNG')
    i = 0
    print("hello")
    for file in png_files:
        strFile = str(file)[:-4]
        print(strFile)
        # Crop image into squares
        img_original = Image.open(file)
        img_crop = expand2square(img_original, (255, 255, 255))
        img_crop = resize_img(250, img_crop)
        img_crop.save(strFile + ".png", quality=95)
        i = i + 1

def resize_img(dimension, image):
    size = dimension, dimension
    im_resized = image.resize(size, Image.ANTIALIAS)
    return im_resized


if __name__ == "__main__":
    #convert_inkml_img()
    imgs2squares()