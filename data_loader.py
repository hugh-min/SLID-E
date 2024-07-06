# coding: utf-8

"""
@File     :data_loader.py
@Author    :XieJing
@Date      :2021/9/23
@Copyright :AI
@Desc      :
"""
import random

import numpy as np
import os
import pandas as pd
import glob
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import sys
from tqdm import tqdm
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# dataset_path = os.path.abspath(os.path.join(project_path,"../","Dataset"))
data_path = os.path.join(project_path, "data")

labels = ["糖网", "黄斑", "青光眼", "正常"]


def get_class_numbers(df):
    class_numbers = {"糖网": list(df["糖网"] == 1).count(True),
                     "黄斑": list(df["黄斑"] == 1).count(True),
                     "青光眼": list(df["青光眼"] == 1).count(True),
                     "正常": list(df["正常"] == 1).count(True)
                     }

    # print(class_numbers)
    return class_numbers


def get_class_resampling_probabilities(df, weight_power=0.65):
    resampling_probabilities = []
    class_numbers = get_class_numbers(df)
    max_numbers = max(list(class_numbers.values()))
    for label in labels:
        if class_numbers[label] != 0:
            prob = (max_numbers ** weight_power) / (class_numbers[label] ** weight_power)
        else:
            raise ValueError("label {} not found".format(label))
        resampling_probabilities.append(prob)
    # print(resampling_probabilities)
    total_ = sum(resampling_probabilities)
    probabilities = np.array(resampling_probabilities) / total_
    # print(probabilities)
    return probabilities


class MyDataset(Dataset):

    def __init__(self, imgs, labels, transform=None, target_transform=None):

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.imgs[idx]
        target = self.labels[idx]

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def train_data_loader(data):
    probabilities = get_class_resampling_probabilities(data)
    df = data
    k = 0
    j = 0
    dataset = {"files": [], "labels": []}
    # k代表总的采样张数，j代表遍历到第几张了，该循环会遍历数据集多轮，每一轮会根据概率从中筛选出一部分图像存在dataset中，直到某一轮
    # k的值等于样本集数量就退出循环。所以有的图像会被重复多次采样，而有的图像则可能一次都不被采样，但是总的采样张数等于数据量
    while True:
        filename = df["filename"][j]
        index = np.where(df.iloc[j, 1:].values == 1)[0]
        max_index = index[0]
        for i in index:
            if probabilities[i] > probabilities[max_index]:
                max_index = i
        p = random.uniform(0, 1)
        if p <= probabilities[max_index]:
            dataset["files"].append(filename)
            dataset["labels"].append(df.iloc[j, 1:].values.tolist())
            k += 1
            if k == len(df):
                break
        if j == len(df) - 1:
            j = 0
        j += 1
    label_arr = np.array(dataset["labels"])
    select_num = np.sum(label_arr, axis=0)
    print(select_num)  # [2663 2577 1453 3164]
    return dataset


class Train_DataSet(Dataset):
    def __init__(self, dataset, H, W):
        self.H = H
        self.W = W
        self.file_list = dataset
        # self.image_size = config.image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.H, self.W)),
                # transforms.CenterCrop(224),
                # transforms.RandomChoice(
                # [transforms.RandomRotation(15),
                #  transforms.ColorJitter(contrast=(0.8, 1.2)),
                #  transforms.RandomHorizontalFlip(),
                #  transforms.RandomVerticalFlip(),
                #  transforms.ColorJitter(brightness=(0.8, 1.2))
                #  ]),

                # transforms.RandomResizedCrop(config.image_size),
                # transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

            ])

    def __len__(self):
        self.filelength = len(self.file_list["files"])
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list["files"][idx]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img)

        label = np.array(self.file_list["labels"][idx])

        return img_transformed, label


class Val_DataSet(Dataset):
    def __init__(self, dataset, H, W):
        self.H = H
        self.W = W
        self.file_list = pd.read_csv(dataset).iloc[:300, :]
        # self.image_size = config.image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.H, self.W)),
                # transforms.Resize(224),
                # transforms.CenterCrop(config.image_size),
                # transforms.RandomChoice(
                #     [transforms.RandomRotation(15),
                #      transforms.ColorJitter(contrast=(0.8, 1.2)),
                #      transforms.RandomHorizontalFlip(),
                #      transforms.RandomVerticalFlip(),
                #      transforms.ColorJitter(brightness=(0.8, 1.2))
                #      ]),

                # transforms.RandomResizedCrop(config.image_size),
                # transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list["filename"][idx]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img)

        label = np.array(self.file_list.iloc[idx, 1:].values.tolist())

        return img_transformed, label


def Normalize(data):
    data = 1 / data
    m = np.sum(data)
    return data / m


if __name__ == '__main__':
    data = pd.read_csv(r"/home/zcb/cy/UWF_6classes/data/train.csv")
    # data_1 = get_class_numbers(data)
    # data_2 = get_class_resampling_probabilities(data)
    data_3 = train_data_loader(data)
    # print("0", data_1)
    # print("1", data_2)
    # print("2", data_3)
    print("3", len(data_3["files"]))

    train_data = Train_DataSet(data_3, 384, 384)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    for images, labels in tqdm(train_loader):
        print(images.shape)
        print(labels)
        break

    val_dataset = Val_DataSet(os.path.join(data_path, "val.csv"), 384, 384)
    for item in val_dataset.file_list["filename"]:
        print(item)
        break

    # label_arr = np.array(datas["labels"])
    # select_num = np.sum(label_arr, axis=0)
    # print(1 / select_num * 1000)
    # print(Normalize(select_num))
