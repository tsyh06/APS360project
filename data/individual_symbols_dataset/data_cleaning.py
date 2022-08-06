import os
import shutil
import PIL.Image
import matplotlib.pyplot as plt
from random import sample, randint
from PIL import Image
import numpy as np

root_dir = '.\\dataset'  # update this to the directory containing data

# create lists of all classes in the root_dir and the number of images in each class
num_classes = len(os.listdir(root_dir))
class_name = [0] * num_classes
class_elem = [0] * num_classes
for i, class_folder in enumerate(os.scandir(root_dir)):
    class_name[i] = class_folder.name
    class_elem[i] = len(os.listdir(class_folder.path))


# function to delete classes from the dataset that are not present in the classes.txt file
def delete_extra_class():
    fo = open('.\\classes.txt', 'r')
    content = fo.readlines()
    class_s = [x.strip() for x in content]
    for class_f in os.scandir(root_dir):
        if class_f.name not in class_s:
            shutil.rmtree(class_f.path)


# function to plot a bar chart of the number of images in each class
def draw_plot():
    plt.title("Bar Chart of the Number of Images in Each Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(range(len(class_name)), class_name)
    plt.xticks(fontsize=6, rotation=90)
    plt.grid(axis='y', color='lightgrey', linewidth=0.5)
    plt.bar(range(len(class_name)), class_elem)
    plt.show()


# function to print all classes with number of images less than a given value
def class_lt(value):
    class_tbd = []
    for j in range(num_classes):
        if class_elem[j] < value:
            class_tbd.append((class_name[j], class_elem[j]))

    print(class_tbd)
    print(len(class_tbd))


# randomly delete excess files from all the classes to trim them down to a value
def random_delete(value):
    for class_f in os.scandir(root_dir):
        num_delete = len(os.listdir(class_f.path)) - value
        if num_delete > 0:
            files = os.listdir(class_f.path)
            for file in sample(files, num_delete):
                os.remove(class_f.path + '\\' + file)


# converting files in a class to png from jpg
def jpg2png(class_path):
    num_files = 1
    for filename in os.listdir(class_path):
        if filename.endswith(".jpg"):
            print(filename)
            filepath = class_path + '\\' + filename  # change class in path
            im = Image.open(filepath)
            rgb_im = im.convert('RGB')
            name = str(num_files) + '.png'
            rgb_im.save(class_path + '\\' + name)
            os.remove(filepath)
            num_files += 1
            print(class_path + '\\' + name)

    print(num_files - 1, "files have been converted from jpg to png for class",
          '\"' + os.path.basename(class_path) + '\"')


# convert all the images in all classes to png
def classes_jpg2png():
    for class_f in os.scandir(root_dir):
        jpg2png(class_f.path)


# expand a specific png to a square
def expand2square(image):
    width, height = image.size  # Get dimensions
    if width == height:
        return image
    else:
        size = max(width, height)
        result = Image.new(image.mode, (size, size), color='white')
        result.paste(image, ((size - width) // 2, (size - height) // 2))
        return result


# resize an image to new dimension
def resize_img(dimension, image):
    size = dimension, dimension
    im_resized = image.resize(size)
    return im_resized


# add white padding on the edges of an image
def expand_borders(image):
    width, height = image.size  # Get dimensions
    size = 70
    result = Image.new(image.mode, (size, size), color='white')
    result.paste(image, ((size - width) // 2, (size - height) // 2))
    return result


# adding white space and resizing images in the given class
def normalize_img(class_path, dimension):
    num_files = 0
    for file in os.scandir(class_path):
        img_original = Image.open(file.path)
        img_crop = expand_borders(img_original)
        img_crop = resize_img(dimension, img_crop)
        img_crop.save(file.path, quality=100)
        num_files += 1

    print(num_files, " files have been resized")


# normalize all classes
def normalize_classes():
    for class_sub in os.scandir(root_dir):
        normalize_img(class_sub.path, 100)


# function to horizontally flip given number of images from a class
def flip_horizontal(class_dir, num_images):
    files = os.listdir(class_dir)
    for file in sample(files, num_images):
        image = Image.open(class_dir + '\\' + file)
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        image.save('.\\new' + '\\' + os.path.basename(class_dir) + '\\' + 'flip' + file)


# function to vertically flip given number of images from a class
def flip_vertical(class_dir, num_images):
    files = os.listdir(class_dir)
    for file in sample(files, num_images):
        image = Image.open(class_dir + '\\' + file)
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        image.save('.\\new' + '\\' + os.path.basename(class_dir) + '\\' + 'flip' + file)


# function to randomly rotate given number of images from a class by an angle within the given range
def rotate_image(class_dir, angle_min, angle_max, num_images):
    files = os.listdir(class_dir)
    for file in sample(files, num_images):
        image = Image.open(class_dir + '\\' + file)
        angle = randint(angle_min, angle_max)
        image = image.rotate(angle=angle, expand=True, fillcolor=(255, 255, 255, 0))
        image = image.resize((100, 100))
        image.save('.\\new' + '\\' + os.path.basename(class_dir) + '\\' + 'rotated_' + str(angle_min) + file)


# function to add salt and pepper noise to given number of images from a class
def add_noise(class_dir, num_images):
    files = os.listdir(class_dir)
    for file in sample(files, num_images):
        image = Image.open(class_dir + '\\' + file)
        image = np.array(image)
        num_pixels = randint(100, 150)
        for k in range(num_pixels):
            # Pick a random y coordinate
            y_coord = randint(0, 99)
            # Pick a random x coordinate
            x_coord = randint(0, 99)

            # Color that pixel to black
            image[y_coord][x_coord] = 0

        num_pixels = randint(100, 150)
        for k in range(num_pixels):
            # Pick a random y coordinate
            y_coord = randint(0, 99)
            # Pick a random x coordinate
            x_coord = randint(0, 99)

            # Color that pixel to white
            image[y_coord][x_coord] = 255

        image = Image.fromarray(image)
        image.save('.\\new' + '\\' + os.path.basename(class_dir) + '\\' + 'noise' + file)


# function to do a horizontal flip and rotate simultaneously on a given number of images from a class
def flip_rotate(class_dir, angle, num_images):
    files = os.listdir(class_dir)
    for file in sample(files, num_images):
        image = Image.open(class_dir + '\\' + file)
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        image = image.rotate(angle=angle, expand=True, fillcolor=(255, 255, 255, 0))
        image = image.resize((100, 100))
        image.save('.\\new' + '\\' + os.path.basename(class_dir) + '\\' + 'flip_rotate' + file)
