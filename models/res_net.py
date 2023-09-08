from torch import nn
from torch.nn import functional as f


# Inception块
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = f.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return f.relu(y)


# 网络结构主体
class ResNet(nn.Module):
    # 允许用户根据数据传入的通道数和输出的特征数动态改变ResNet结构，提高代码的可复用性
    def __init__(self, input_channels, out_features):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(), nn.Linear(512, out_features))

    def forward(self, x):
        x = self.net(x)
        return x


class MiniResNet(nn.Module):
    # 允许用户根据数据传入的通道数和输出的特征数动态改变ResNet结构，提高代码的可复用性
    def __init__(self, input_channels, out_features):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(32), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(32, 32, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(32, 128, 2))
        self.net = nn.Sequential(b1, b2, b3, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, out_features))

    def forward(self, x):
        x = self.net(x)
        return x
