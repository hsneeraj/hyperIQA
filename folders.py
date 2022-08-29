import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import pandas as pd
import fnmatch
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, label=0):
        self.img_dir = img_dir
        self.transform = transform
        self.label = label
        self.items = fnmatch.filter(os.listdir(img_dir), '*.png')
        self.LEN = len(self.items)
        # self.images = [None] * self.LEN
        print('Creating dataset from', img_dir, 'with length', self.LEN)

    def __len__(self):
        return self.LEN

    def __getitem__(self, idx):
        print('       not found index', idx)
        img_path = self.img_dir + self.items[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return (image, self.label)

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename
