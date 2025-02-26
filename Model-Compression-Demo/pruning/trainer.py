import torch  # 导入PyTorch库
from torchvision import datasets, transforms  # 导入数据集和预处理工具
from torch.utils.data import DataLoader  # 导入数据加载器
import math  # 导入数学库，用于计算进度条长度
import sys  # 导入sys库用于系统路径操作

sys.path.append(r'E:\desktop\Model-Compression-Demo-master')  # 添加项目特定路径到系统路径

# 从自定义模块pruning.nets中导入MyNet类，即要训练的网络模型
from pruning.nets import MyNet


class Trainer:
    """定义一个训练器类，负责训练模型并保存训练后的权重"""

    def __init__(self, save_path):
        """初始化方法，设置设备、模型、数据集、优化器等"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 设置运行设备为GPU或CPU
        self.save_path = save_path  # 模型保存路径
        self.net = MyNet().to(self.device)  # 实例化要训练的网络模型，并移动到指定设备上

        # 定义图像转换，包括转换为张量和标准化
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 对于MNIST，这里使用均值和标准差都是0.5进行标准化
        ])

        # 加载MNIST训练数据集，并创建DataLoader实例
        self.train_data = DataLoader(
            datasets.MNIST(r"E:\点头第五期课程\模型剪枝和量化和蒸馏\Model-Compression-Demo-master\datasets",
                           train=True, transform=self.trans, download=False),
            batch_size=100, shuffle=True, drop_last=True  # 设置batch size、是否打乱顺序以及是否丢弃最后一个不完整的batch
        )

        # 初始化Adam优化器
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.net.train()  # 将模型设置为训练模式

    def train(self):
        """执行模型训练过程"""
        for epoch in range(1, 3):  # 训练两个epoch
            total = 0  # 用于累计处理的数据点数量
            for i, (data, label) in enumerate(self.train_data):  # 遍历数据加载器
                data, label = data.to(self.device), label.to(self.device)  # 将数据移动到指定设备上

                output = self.net(data)  # 前向传播，获取模型输出
                loss = self.net.get_loss(output, label)  # 计算损失函数

                self.optimizer.zero_grad()  # 清空之前的梯度
                loss.backward()  # 反向传播计算梯度
                self.optimizer.step()  # 更新模型参数

                total += len(data)  # 累计处理的数据点数量
                progress = math.ceil(i / len(self.train_data) * 50)  # 计算进度条长度
                print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')  # 打印进度条

            # 在每个epoch结束时保存模型状态字典
            torch.save(self.net.state_dict(), self.save_path)


if __name__ == '__main__':
    # 创建Trainer实例并开始训练
    trainer = Trainer("models/net.pth")
    trainer.train()