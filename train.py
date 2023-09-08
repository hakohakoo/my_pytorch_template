import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

import global_args
from global_args import device

if __name__ == "__main__":
    # 设置参数集
    args = global_args.PositionNet01Args

    # 数据集读取
    data, labels = args.get_data_and_labels(for_test=False)

    # 训练集和验证集的划分
    dataset = TensorDataset(data, labels)
    train_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[1 - args.val_fraction, args.val_fraction],
        generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    # 存储并展示loss数值
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    # 一个简单的反向传播训练流程
    # 使用更复杂网络时需要修改下列代码
    for epoch in range(args.epochs):
        # ========================== train step ==========================
        args.model.train()
        train_epoch_loss = []
        for idx, (train_data, train_labels) in enumerate(train_loader, 0):
            train_data = train_data.to(torch.float32).to(device)
            train_labels = train_labels.to(torch.float32).to(device)
            outputs = args.model(train_data)

            args.optimizer.zero_grad()
            loss = args.criterion(train_labels, outputs)
            loss.backward()
            args.optimizer.step()

            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
        train_epochs_loss.append(np.average(train_epoch_loss))

        # valid step 用于监控对比train loss和valid loss，确保两个loss同时下降，以防止梯度爆炸或者过拟合的情况
        # 两种loss的分析可参考：https://blog.csdn.net/qq_44866009/article/details/122274263
        # ========================== valid step ==========================
        args.model.eval()
        valid_epoch_loss = []
        for idx, (val_data, val_labels) in enumerate(val_loader, 0):
            val_data = val_data.to(torch.float32).to(device)
            val_labels = val_labels.to(torch.float32).to(device)
            outputs = args.model(val_data)

            loss = args.criterion(outputs, val_labels)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))

        print(f"epoch={epoch}/{args.epochs}, train_loss:{train_epochs_loss[epoch]} val_loss:{valid_epochs_loss[epoch]}")

        # 保存模型
        if epoch % args.save_period == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(args.model.state_dict(), f'{args.save_path}/model_{epoch}.pt')

# 生成训练过程中两种loss的折线图
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(train_loss[:])
plt.title("train_loss")
plt.subplot(122)
plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
plt.title("epochs_loss")
plt.legend()
plt.show()
