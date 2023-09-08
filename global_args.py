import os
import torch
import models.res_net
from torch import optim
from utils.file_utils import load_data_and_labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
proj_path = os.path.abspath(os.path.dirname(__file__))


# 使用静态类储存一组参数，每一组参数对应一个静态类
class PositionNet01Args:
    proj_name = "resnet_for_positioning01"

    # 传入Resnet的两个参数分别表示：输入的数据为单通道 且最终模型需要3个输出数值
    model = models.res_net.ResNet(1, 3).to(device)  # 使用的网络结构模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器和学习率设置
    criterion = torch.nn.MSELoss()  # loss 函数
    batch_size = 128
    epochs = 50  # 训练轮数
    val_fraction = 0.2  # 划分多少数据作为训练时的验证集，这里是20%

    save_path = os.path.join(proj_path, f'checkpoints/{proj_name}')  # 训练模型的保存路径
    save_period = 5  # 设置每隔几个epoch保存一次

    # 一个模型对应一组数据和一套数据预处理方法
    @staticmethod
    def get_data_and_labels(for_test=False):
        # 数据读取部分
        data_path = os.path.join(proj_path, 'datasets/train_1024_1_circle_pos/train/data')
        labels_path = os.path.join(proj_path, 'datasets/train_1024_1_circle_pos/train/label')
        if for_test:
            data_path = os.path.join(proj_path, 'datasets/train_1024_1_circle_pos/test/data')
            labels_path = os.path.join(proj_path, 'datasets/train_1024_1_circle_pos/test/label')

        # label_index用于标记选取label的哪几个数值，这里的0 1 2代表，仅预测水下目标物的x y z坐标（而忽略label文件中保存的磁化率，长宽高）
        data, labels = load_data_and_labels(data_dir_path=data_path, label_dir_path=labels_path,
                                                label_index=(0, 1, 2))
        # 数据预处理部分
        data = data.reshape((-1, 1, 32, 32))  # 注意reshape格式
        return torch.from_numpy(data), torch.from_numpy(labels)  # 输出成为torch tensor
