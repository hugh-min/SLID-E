import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class KeNet(nn.Module):
    def __init__(self, classes_num):
        super(KeNet, self).__init__()
        resNet = models.resnet50(pretrained=True)
        resNet = list(resNet.children())[:-2]  # 2048 * 7 * 7
        self.features = nn.Sequential(*resNet)

        self.attention = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.Conv2d(2048, 64, kernel_size=1, padding=0),  # 64 * 7 * 7
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1, padding=0),  # 16 * 7 * 7
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1, padding=0),  # 8 * 7 * 7
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1, padding=0),  # 1 * 7 * 7
            nn.Sigmoid()
        )
        self.up_c2 = nn.Conv2d(1, 2048, kernel_size=1, padding=0, bias=False)
        nn.init.constant_(self.up_c2.weight, 1)
        self.denses = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Dropout(0.5),
            nn.Linear(256, classes_num)
        )

    def forward(self, x):
        x = self.features(x)  # 2048 * 7 * 7

        atten_layers = self.attention(x)  # 1 * 7 * 7
        atten_layers = self.up_c2(atten_layers)  # 2048 * 7 * 7
        # print atten_layers.shape
        mask_features = torch.matmul(atten_layers, x)  # 2048 * 7 * 7

        batch, channels, height, width = atten_layers.size()
        if torch.is_tensor(height):
            height = height.item()  # 这里是修正代码
            width = width.item()  # 这里是修正代码

        # print mask_features.shape
        gap_features = F.avg_pool2d(mask_features, kernel_size=[height, width]).view(batch, -1)  # 2048 * 1 * 1
        # print gap_features.shape
        gap_mask = F.avg_pool2d(atten_layers, kernel_size=[height, width]).view(batch, -1)  # 2048 * 1 * 1
        # gap_mask = F.avg_pool2d(atten_layers, kernel_size=atten_layers.size()[2:])
        # print gap_mask.shape
        gap = Lambda(lambda x: x[0] / x[1])([gap_features, gap_mask])  # 2048 * 1 * 1
        # print gap.shape
        x = self.denses(gap) # num_classes * 7 * 7
        # x = torch.unsqueeze(x, 0)
        return x