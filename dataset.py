import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging
from PIL import Image
import cv2
from utils import scale_to_dst_size


def preprocess(img, dstSize, transform=None):
    if transform:
        # img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform(img)  # 用transform进行各种变换

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    img_scaled = scale_to_dst_size(img, dstSize)

    # cv2.imwrite('test.jpg', img_scaled)

    # HWC to CHW
    img_trans = img_scaled.transpose((2, 0, 1))
    img_trans = img_trans / 255

    return img_trans


def preprocess1(img, dstSize, transform=None):
    results = []
    if transform:
        # img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform(img)  # 用transform进行各种变换

    for img_sub in img:
        img_sub = cv2.cvtColor(np.asarray(img_sub), cv2.COLOR_RGB2BGR)

        img_scaled = scale_to_dst_size(img_sub, dstSize)

        # cv2.imwrite('test.jpg', img_scaled)

        # HWC to CHW
        img_trans = img_scaled.transpose((2, 0, 1))
        img_trans = img_trans / 255

        results.append(img_trans)

    return results


class MyDataset(Dataset):
    def __init__(self, imgs_dir, dstSize, transform=None, labels=None):
        self.dstSize = dstSize

        categories = os.listdir(imgs_dir) if labels is None else labels
        self.img_path = []
        self.ids = []
        self.labels = []
        for i in range(len(categories)):
            category_dir = os.path.join(imgs_dir, categories[i])
            imgs = os.listdir(category_dir)
            self.labels.append(categories[i])
            for img in imgs:
                self.img_path.append(os.path.join(category_dir, img))
                self.ids.append(i)

        self.transform = transform

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def get_labels(self):
        return self.labels

    def __getitem__(self, i):
        img_path = self.img_path[i]
        id = self.ids[i]

        # id = float(self.labels[id])
        # id = torch.Tensor(id, dtype=torch.float32)

        # img = cv2.imread(img_path)
        img = Image.open(img_path)
        img_trans = preprocess(img, self.dstSize, self.transform)
        img_trans = torch.Tensor(img_trans)

        return img_trans, id