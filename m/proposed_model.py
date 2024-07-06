import torch
from torch import nn
from torch import tensor
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # input[3, 224, 224]  output[64, 112, 112]
            nn.BatchNorm2d(64),  # input[64, 224, 224]  output[64, 112, 112]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # input[64, 224, 224]  output[64, 56, 56]
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),  # input[64, 56, 56]  output[64, 56, 56]
            nn.BatchNorm2d(64),  # input[64, 56, 56]  output[64, 56, 56]
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),  # input[64, 56, 56]  output[192, 56, 56]
            nn.BatchNorm2d(192),  # input[192, 56, 56]  output[192, 56, 56]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # input[192, 56, 56]  output[192, 28, 28]
            Inception(192, 64, 96, 128, 16, 32, 32),  # input[192, 28, 28]  output[256, 28, 28]
            Inception(256, 128, 128, 192, 32, 96, 64),  # input[256, 28, 28]  output[480, 28, 28]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # input[480, 28, 28]  output[480, 14, 14]
            Inception(480, 192, 96, 208, 16, 48, 64),  # input[480, 14, 14]  output[512, 14, 14]
            Inception(512, 160, 112, 224, 24, 64, 64),  # input[512, 14, 14]  output[512, 14, 14]
            Inception(512, 128, 128, 256, 24, 64, 64),  # input[512, 14, 14]  output[512, 14, 14]
            Inception(512, 112, 144, 288, 32, 64, 64)  # input[512, 14, 14]  output[528, 14, 14]
        )

    def forward(self, x):
        return self.head(x)

class MaxFeatureFusion(nn.Module):
    def __init__(self):
        super(MaxFeatureFusion, self).__init__()

    def forward(self, x):
        # f1, f2, f3, f4 = x[0], x[1], x[2], x[3]
        # batch, H, W = f1.shape[0], f1.shape[2], f1.shape[3]
        # channel = f1.shape[1]
        # batch_max_feature = []
        # for b in range(batch):
        #     c_max = []
        #     for c in range(channel):
        #         h_max = []
        #         for h in range(H):
        #             w_max, tem_w = [], 0
        #             for w in range(W):
        #                 tem_w = max(f1[b][c][h][w], f2[b][c][h][w], f3[b][c][h][w], f4[b][c][h][w])
        #                 w_max.append(tem_w)
        #             h_max.append(w_max)
        #         c_max.append(h_max)
        #     batch_max_feature.append(c_max)
        # batch_max_feature = tensor(batch_max_feature)

        f1, f2, f3, f4 = x[0], x[1], x[2], x[3]
        batch, H, W = f1.shape[0], f1.shape[2], f1.shape[3]
        channel = f1.shape[1]
        max_features = []
        for b in range(batch):
            batch_max_features = []
            for c in range(channel):
                channel_max_features = []
                for f in range(4):
                    channel_max_features.append(x[f][b][c].cpu().detach().numpy())
                channel_max_features = np.max(channel_max_features, axis=0)
                batch_max_features.append(channel_max_features)
            max_features.append(batch_max_features)
        max_features = tensor(max_features)

        return max_features.to(device)

class MaxFeatureFusion(nn.Module):
    def __init__(self):
        super(MaxFeatureFusion, self).__init__()

    def forward(self, x):
        # f1, f2, f3, f4 = x[0], x[1], x[2], x[3]
        # batch, H, W = f1.shape[0], f1.shape[2], f1.shape[3]
        # channel = f1.shape[1]
        # batch_max_feature = []
        # for b in range(batch):
        #     c_max = []
        #     for c in range(channel):
        #         h_max = []
        #         for h in range(H):
        #             w_max, tem_w = [], 0
        #             for w in range(W):
        #                 tem_w = max(f1[b][c][h][w], f2[b][c][h][w], f3[b][c][h][w], f4[b][c][h][w])
        #                 w_max.append(tem_w)
        #             h_max.append(w_max)
        #         c_max.append(h_max)
        #     batch_max_feature.append(c_max)
        # batch_max_feature = tensor(batch_max_feature)

        f1, f2, f3, f4 = x[0], x[1], x[2], x[3]
        batch, H, W = f1.shape[0], f1.shape[2], f1.shape[3]
        channel = f1.shape[1]
        max_features = []
        for b in range(batch):
            batch_max_features = []
            for c in range(channel):
                channel_max_features = []
                for f in range(4):
                    channel_max_features.append(x[f][b][c].cpu().detach().numpy())
                channel_max_features = np.max(channel_max_features, axis=0)
                batch_max_features.append(channel_max_features)
            max_features.append(batch_max_features)
        max_features = tensor(max_features)

        return max_features.to(device)


class ProposedModel(nn.Module):
    def __init__(self, num_classes=4, init_weights=True, **kwargs):
        super(ProposedModel, self).__init__()
        self.features1 = Head()
        self.features2 = Head()
        self.features3 = Head()
        self.features4 = Head()
        self.feature_fusion = MaxFeatureFusion()
        self.inception1 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.inception2 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception3 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.features1(x[0])
        x2 = self.features2(x[1])
        x3 = self.features3(x[2])
        x4 = self.features4(x[3])
        x = self.feature_fusion([x1, x2, x3, x4])
        x = self.inception1(x)
        x = self.maxpool1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
