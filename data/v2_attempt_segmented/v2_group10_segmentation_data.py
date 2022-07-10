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

    csp = str(rootdirectory)
    fullpath = csp + "/" + txt_file

    num_classes = len(class_dictionary)
    num_dataset = sum(class_dictionary.values())
    f = open(fullpath, "x")
    f = open(fullpath, "w")
    f.write(str(num_classes))
    f.write(" Classes")
    f.write("-----------------------------------------------------\n")
    width_col1 = 10
    width_col2 = 4
    count = 0
    for key, value in class_dictionary.items():
        f.write(key)
        for i in range(width_col1-len(key)):
            f.write(' ')
        f.write("|")
        for i in range(width_col2-len(str(value))):
            f.write(' ')
        f.write(str(value))
        f.write('|')
        for i in range(width_col2-len(str(value))):
            f.write(' ')
        f.write(str(value/num_dataset))
        f.write("\n")
        count += 1
        if count % 5 == 0:
            f.write("\n")
    f.close()


if __name__ == "__main__":
    root_dir = Path(os.getcwd())
    class_dir = Path(os.getcwd() + "/classes")
    class_list, class_size_list, class_dictionary = print_class_stats(class_dir)
    print_summary_stats(root_dir, class_list, class_size_list, class_dictionary)
    print(sum(class_dictionary.values()))
    #string = "MAEESHA BISWAS"
    #rint(string[:-3])