from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def get_model(n_classes, pretrained):
    model = models.resnet50(pretrained=pretrained)
    fc_input = model.fc.in_features
    model.fc = nn.Linear(fc_input, n_classes)  # 修改最后一层全连接的输出类型

    # model.fc = nn.Linear(fc_input, 1)  # 修改最后一层全连接的输出类型

    return model


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernelSize, dilation=1):
        super(BasicBlock, self).__init__()
        padding = ((kernelSize[0] - 1) // 2 * dilation, (kernelSize[1] - 1) // 2 * dilation)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernelSize, padding=padding, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernelSize, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        # 原x要与经过处理后的x的维度相同
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


# channels为BasicBlock的通道数，例如[[3, 64], [64, 64]]表示有两个BasicBlock，其输入输出分别为[3, 64]和[64, 64]
def block_layer(block, channels, kernelSize, dilation=1):
    layers = []
    for i in range(len(channels)):
        layers.append(block(channels[i][0], channels[i][1], kernelSize=kernelSize, dilation=dilation))

    return nn.Sequential(*layers)


class LineNet(nn.Module):
    def __init__(self, n_classes, img_size):
        super(LineNet, self).__init__()
        self.n_classes = n_classes
        b_channel = 32  # base_channel
        kernel_size = (3, 3)  # 分别表示x方向和y方向的卷积长度
        # 卷积提取特征，生成通道为1，与原图大小一样的特征图

        self.conv1 = block_layer(BasicBlock, [[3, b_channel * 2]], kernel_size)
        # self.conv1 = block_layer(BasicBlock, [[3, b_channel * 2], [b_channel * 2, b_channel * 4]], kernel_size)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = block_layer(BasicBlock, [[b_channel * 2, 1]], kernel_size)
        # self.conv2 = block_layer(BasicBlock, [[b_channel * 4, b_channel * 2], [b_channel * 2, 1]], kernel_size)

        # 传入lstm
        lstm_hidden_units = 256
        self.lstm = nn.LSTM(256, lstm_hidden_units, bidirectional=False)
        # self.linear1 = nn.Linear(lstm_hidden_units * 2, lstm_hidden_units)
        self.linear2 = nn.Linear(lstm_hidden_units, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        out = self.conv2(x)

        # out_np = out.detach().cpu().numpy()
        # _range = np.max(out_np) - np.min(out_np)
        # out_np = (out_np - np.min(out_np)) / _range
        # out_np = np.array(out_np * 255, dtype=np.uint8)
        # cv2.imwrite('feature_map.jpg', out_np[0][0])

        b, c, h, w = out.size()
        out = out.permute(3, 0, 1, 2)
        out = out.contiguous().view(w, b, c * h)

        out, _ = self.lstm(out)

        # 获取最后一步的输出
        out_last = out[-1, :, :]

        # out_last = self.linear1(out_last)
        out_last = self.linear2(out_last)
        # out_last = out_last.view(b, self.n_classes, -1)
        # out_last = self.softmax(out_last)

        return out_last


class LineNet_cnn(nn.Module):
    def __init__(self, n_classes, img_size):
        super(LineNet_cnn, self).__init__()
        self.n_classes = n_classes
        b_channel = 32  # base_channel
        kernel_size = (3, 3)  # 分别表示x方向和y方向的卷积长度
        # 卷积提取特征，生成通道为1，与原图大小一样的特征图

        self.conv1 = block_layer(BasicBlock, [[3, b_channel * 2]], kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = block_layer(BasicBlock, [[b_channel * 2, 1]], kernel_size)

        # 对每行进行卷积
        w = img_size[0] // 2
        h = img_size[1] // 2
        self.line_conv = nn.Conv2d(h, h, kernel_size=(1, w), groups=h)  # 分组卷积
        self.linear = nn.Linear(h, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        out = self.conv2(x)

        # 先将1个通道的特征图转换为通道数为h，尺寸1*w的特征图
        b, c, h, w = out.size()
        out = out.contiguous().view(b, c * h, 1, w)

        out = self.line_conv(out)
        out = out.view(b, h)
        out = self.linear(out)

        return out


class LineNet_cnn1(nn.Module):
    def __init__(self, n_classes, img_size):
        super(LineNet_cnn1, self).__init__()
        self.n_classes = n_classes
        b_channel = 32  # base_channel
        kernel_size = (3, 3)  # 分别表示x方向和y方向的卷积长度
        # 卷积提取特征，生成通道为1，与原图大小一样的特征图

        self.conv1 = block_layer(BasicBlock, [[3, b_channel * 2], [b_channel * 2, b_channel * 4]], kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = block_layer(BasicBlock, [[b_channel * 4, b_channel * 4], [b_channel * 4, b_channel * 4]], kernel_size)

        # 对每行进行卷积
        w = img_size[0] // 2
        h = img_size[1] // 2
        # self.line_conv = nn.Conv2d(b_channel * 4, 1, kernel_size=(1, w))
        # self.linear = nn.Linear(h, n_classes)
        self.line_conv = nn.Conv2d(b_channel * 4, 1, kernel_size=(2, w), stride=(2, 1))
        self.linear = nn.Linear(h // 2, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        out = self.conv2(x)

        out = self.line_conv(out)

        # 先将1个通道的特征图转换为通道数为h，尺寸1*w的特征图
        b, c, h, w = out.size()
        out = out.view(b, h)
        out = self.linear(out)

        return out
