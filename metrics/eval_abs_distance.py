import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt, rcParams
from torch import Tensor

import global_args

# 本文件测试了一次训练中所有保存下来的模型的误差分布
# 生成的折线图中某单条折线上的一点表示：当前模型n在整个测试集上有占比y的数据误差小于x

# 设置参数集并读取模型
args = global_args.PositionNet01Args
test_data, test_labels = args.get_data_and_labels(for_test=True)
# 注意后面的to(torch.float32).to(global_args.device)，需要与训练过程中的一致
test_data = test_data.to(torch.float32).to(global_args.device)
test_labels = test_labels.to(torch.float32).to(global_args.device)
args.model.eval()

# 出图的参数
inspection_range = 100  # x轴范围
inspection_lines = 10  # 选出效果最好的10个模型
config = {
    "font.family": 'serif',
    "font.size": 16,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
    "axes.unicode_minus": False,
}
rcParams.update(config)

results = []
results_x = []
results_y = []
results_z = []
for model_path in os.listdir(args.save_path):
    # 读取模型
    args.model.load_state_dict(torch.load(os.path.join(args.save_path, model_path)))
    predict_labels = args.model(test_data)

    result = []
    result_ratio = []
    result_x, result_y, result_z = [], [], []
    result_ratio_x, result_ratio_y, result_ratio_z = [], [], []
    for diff in predict_labels - test_labels:
        # 此处将pytorch的tensor转化为numpy不是必要
        # 可以始终使用pytorch的tensor做处理和出图
        diff = Tensor.cpu(diff).detach().numpy()
        result.append(math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2))
        result_x.append(abs(diff[0]))
        result_y.append(abs(diff[1]))
        result_z.append(abs(diff[2]))

    result_ratio = np.array(result[:])
    result_ratio_x = np.array(result_x[:])
    result_ratio_y = np.array(result_y[:])
    result_ratio_z = np.array(result_z[:])

    result_ratio = [np.sum((result_ratio < i) != 0) / result_ratio.shape[0] for i in range(inspection_range)]
    result_ratio_x = [np.sum((result_ratio_x < i) != 0) / result_ratio_x.shape[0] for i in range(inspection_range)]
    result_ratio_y = [np.sum((result_ratio_y < i) != 0) / result_ratio_y.shape[0] for i in range(inspection_range)]
    result_ratio_z = [np.sum((result_ratio_z < i) != 0) / result_ratio_z.shape[0] for i in range(inspection_range)]

    results.append((result_ratio, model_path))
    results_x.append((result_ratio_x, model_path))
    results_y.append((result_ratio_y, model_path))
    results_z.append((result_ratio_z, model_path))


name = ['绝对误差分布', 'X方向绝对误差分布', 'Y方向绝对误差分布', 'Z方向绝对误差分布']
for idx, cur_result in enumerate([results, results_x, results_y, results_z]):
    for diff in range(len(cur_result) - 1):
        for j in range(len(cur_result) - 1 - diff):
            if cur_result[j][0][39] > cur_result[j + 1][0][39]:
                temp = cur_result[j + 1]
                cur_result[j + 1] = cur_result[j]
                cur_result[j] = temp

    plt.figure(figsize=(8, 8))
    for diff in range(inspection_lines):
        plt.plot([i for i in range(inspection_range)], cur_result[len(cur_result) - 1 - diff][0],
                 label=cur_result[len(cur_result) - 1 - diff][1])
    plt.legend(loc="lower right")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(name[idx], fontsize=28)
    plt.ylabel("数据占比", fontsize=20)
    plt.xlabel("绝对误差 (m)", fontsize=20)
    plt.grid()
    plt.show()
