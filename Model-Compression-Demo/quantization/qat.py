# 导入必要的PyTorch库和其他依赖项
import torch
from torch import nn
import sys

sys.path.append(r'E:\desktop\Model-Compression-Demo-master')  # 添加项目特定路径到系统路径

# 从自定义模块导入网络模型MyNet
from quantization.nets import MyNet
from torchvision import datasets, transforms  # 导入数据集和预处理工具
from torch.utils.data import DataLoader  # 导入数据加载器
import math  # 导入数学库，用于计算进度条
import os  # 导入操作系统库，用于文件操作

# 再次导入已经导入过的库（这行是冗余的，通常不需要）
import torch
from torch import nn
from quantization.nets import MyNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os

"""
1.设置量化配置：
在模型中指定哪些层或模块需要进行量化，并选择合适的量化方案（例如，8位整数量化）。
这通常涉及到为模型分配一个 qconfig，该配置定义了量化的规则和参数。
2.准备量化感知训练：
使用 torch.quantization.prepare_qat(model, inplace=True) 准备模型进行 QAT。这个函数会将量化存根（quant stubs）插入到模型中，以便在前向传递时模拟量化行为。
它还会融合某些常见的层组合（如 Conv + BN + ReLU），这样可以在量化后保持性能。
3.训练模型：
模型现在应该处于训练模式 (model.train())，并像平常一样进行训练。在这个阶段，模型不仅学习权重，还学习如何适应量化带来的限制。
由于量化模拟是可微分的，所以可以使用标准的优化器和损失函数来进行训练。
4.转换为量化模型：
训练完成后，将模型切换到评估模式 (model.eval()) 并调用 torch.quantization.convert(model, inplace=True) 将其转换为实际的量化模型。
这个过程会移除量化存根，并替换为真实的量化操作。
5.验证量化模型：
使用测试集或验证集对量化后的模型进行评估，确保其性能符合预期。
6.保存量化模型：
最后，保存量化后的模型以供部署使用。
"""

class QAT:
    def __init__(self, net_path):
        """
        初始化QAT类，设置设备、加载模型、准备训练数据等。
        :param net_path: 模型权重的路径。
        """
        self.device = "cpu" if torch.cuda.is_available() else "cpu"  # 设置运行设备为GPU或CPU
        self.net = MyNet().to(self.device)  # 实例化网络模型，并移动到指定设备上
        self.net.load_state_dict(torch.load(net_path, map_location=self.device))  # 加载模型权重
        self.net.eval()  # 将模型设置为评估模式，但稍后会切换回训练模式
        self.net.fuse_model()  # 融合模型中的某些操作以优化量化

        # 定义图像转换，包括转换为张量和标准化
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 加载MNIST训练数据集，并创建DataLoader实例
        self.train_data = DataLoader(
            datasets.MNIST("../datasets/", train=True, transform=self.trans, download=False),
            batch_size=100, shuffle=True, drop_last=True)

        self.optimizer = torch.optim.Adam(self.net.parameters())  # 定义Adam优化器
        self.net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')  # 设置默认量化配置，使用fbgemm后端

    def prepare_for_qat(self):
        """
        准备网络进行量化感知训练。
        """
        self.net.train()  # 确保模型处于训练模式
        torch.quantization.prepare_qat(self.net, inplace=True)  # 准备量化感知训练

    def train(self):
        """
        执行量化感知训练（QAT），并保存量化后的模型。
        """
        # torch.quantization.prepare_qat(self.net, inplace=True)  # 准备量化感知训练，融合量化模拟与模型
        self.prepare_for_qat()  # 准备量化感知训练
        for epoch in range(1, 3):  # 进行2个epoch的训练
            total = 0  # 记录已处理的数据总量
            for i, (data, label) in enumerate(self.train_data):  # 遍历训练数据集
                data, label = data.to(self.device), label.to(self.device)  # 将数据移动到指定设备上

                output = self.net(data)  # 前向传播，获取预测输出
                loss = self.net.get_loss(output, label)  # 计算损失值

                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播，计算梯度
                self.optimizer.step()  # 更新参数

                total += len(data)  # 累加处理的数据量
                progress = math.ceil(i / len(self.train_data) * 50)  # 计算进度条长度
                print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')  # 打印进度信息

        # 在每个epoch结束后，将模型转换为量化模型，并检查其准确性
        self.net = torch.quantization.convert(self.net.eval(), inplace=False)  # 将模型转换为量化模型

        # 保存量化后的模型为TorchScript格式
        torch.jit.save(torch.jit.script(self.net), "models/net_convert_qat.pth")

        # 打印模型大小（以MB为单位）
        print('\nSize (MB):', os.path.getsize("models/net_convert.pth") / 1e6)


if __name__ == '__main__':
    qat = QAT("models/net.pth")  # 创建QAT实例，加载原始模型
    qat.train()  # 开始量化感知训练过程