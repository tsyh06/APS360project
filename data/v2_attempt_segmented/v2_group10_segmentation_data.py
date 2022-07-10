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
import fnmatch
from datetime import datetime
import re
import shutil
import os
# Remod DIR from the ends of class names


# original set of 61 classes from https://researchdata.edu.au/crohme-competition-recognition-expressions-png/639782
# continuing to augment
def produce_stats_txt():
    directory = "./classes/" # Change this to wherever classes are
    for count, filename in enumerate(os.listdir(directory)):
        oldname = directory + filename
        print(oldname)
        newname = str(oldname)[:-3]
        print(newname)
        os.rename(oldname, newname)


def print_class_stats(classdirectory):
    # rootdir = Path(os.getcwd() + "/classes")

    class_size_list = []
    class_list = []
    i = 0
    for file in os.listdir(classdirectory):
        d = os.path.join(classdirectory, file)
        if os.path.isdir(d):
            num_files = len(fnmatch.filter(os.listdir(d), '*.png'))
            class_size_list.append(num_files)
            # print(num_files)
            i += 1
    for count, filename in enumerate(os.listdir(classdirectory)):
        # print(filename)
        class_list.append(filename)

    class_dictionary = {class_list[i]: class_size_list[i] for i in range(len(class_list))}
    # class_dictionary = dict(zip(class_list, class_size_list))
    return class_list, class_size_list, class_dictionary

def print_summary_stats(rootdirectory, class_list, class_size_list, class_dictionary):
    txt_file = 'stats_' + str(datetime.now().strftime('%Y_%m_%d--%H_%M_%S')) + '.txt'
    class_txt_file = 'classes.txt'

    csp = str(rootdirectory)
    fullpath = csp + "/" + txt_file
    path_classes_file = csp + "/" + class_txt_file

    num_classes = len(class_dictionary)
    num_dataset = sum(class_dictionary.values())
    f = open(fullpath, "x")
    f = open(fullpath, "w")

    f2 = open(path_classes_file, "w")
    f.write(str(num_classes))
    f.write(" Classes\n")
    f2.write(str(num_classes))
    f2.write(" Classes\n")
    f.write("Class------Number-----%------------------------------\n")
    width_col1 = 10
    width_col2 = 10
    count = 0
    for key, value in class_dictionary.items():
        f.write(key)
        f2.write(key)
        f2.write("\n")
        for i in range(width_col1-len(key)):
            f.write(' ')
        f.write("|")
        for i in range(width_col2-len(str(value))):
            f.write(' ')
        f.write(str(value))
        f.write('|    ')
        #for i in range(width_col2-len(str(value))):
        #    f.write(' ')
        f.write(str(round(value/num_dataset,5)))
        f.write("\n")
        count += 1
        #if count % 5 == 0:
        #    f.write("\n")
    f.close()
    plt.title("Bar Chart of the Number of Images in Each Class Before Data Augmentation")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(range(len(class_list)), class_list)
    plt.bar(range(len(class_list)), class_size_list)
    plt.show()

if __name__ == "__main__":
    root_dir = Path(os.getcwd())
    class_dir = Path(os.getcwd() + "/classes")
    class_list, class_size_list, class_dictionary = print_class_stats(class_dir)
    print_summary_stats(root_dir, class_list, class_size_list, class_dictionary)
    #print(sum(class_dictionary.values()))

    source = Path(os.getcwd() + "/classes/times")
    destination = "Desktop/content/"

    # code to move the files from sub-folder to main folder.
    #files = os.listdir(source)
    #print(files)
    #for file in files:
    #    file_name = os.path.join(source, file)
    #    print(file_name)
    #    data_files = os.listdir(file_name)
    #     for data in data_files:
    #         data_file_name = os.path.join(file_name, data)
    #         from pathlib import Path
    #        #print(data[-4:])
    #        if data[-4:] == '.png':
    #             print(data_file_name)
    #            shutil.copy(data_file_name, source)