import os
import json
import shutil

import cv2
import numpy as np
from utils import scale_to_dst_size


def create_dataset():
    src_path = '//data/'
    folders = os.listdir(src_path + 'src')

    for folder in folders:
        if not os.path.exists(src_path + 'train/' + folder):
            os.mkdir(src_path + 'train/' + folder)

        if not os.path.exists(src_path + 'val/' + folder):
            os.mkdir(src_path + 'val/' + folder)

        files = os.listdir(src_path + 'src/' + folder)
        count = 0
        for file in files:
            count += 1
            if count % 6 == 0:
                shutil.copyfile(src_path + 'src/' + folder + '/' + file, src_path + 'val/' + folder + '/' + file)
            else:
                shutil.copyfile(src_path + 'src/' + folder + '/' + file, src_path + 'train/' + folder + '/' + file)


if __name__ == "__main__":
    create_dataset()