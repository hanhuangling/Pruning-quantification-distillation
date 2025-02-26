import torch
from torchvision import datasets, transforms  # 导入数据集和预处理模块
from torch.utils.data import DataLoader  # 导入DataLoader用于加载数据
import math  # 导入数学库，用于计算进度条
import sys  # 导入系统库，用于路径操作
sys.path.append(r'E:\desktop\Model-Compression-Demo-master')  # 添加项目特定路径到系统路径
from quantization.nets import MyNet  # 从指定路径导入自定义网络模型MyNet


class Trainer:
    def __init__(self, save_path):
        """
        初始化Trainer类，设置设备、加载数据集、创建模型和优化器。
        :param save_path: 模型保存路径。
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 设置运行设备为GPU或CPU
        self.save_path = save_path  # 设置模型保存路径
        self.net = MyNet().to(self.device)  # 实例化网络模型，并移动到指定设备上
        self.trans = transforms.Compose([  # 定义图像预处理转换
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor
            transforms.Normalize([0.5], [0.5])  # 归一化图像张量，使均值为0.5，标准差为0.5
        ])
        self.train_data = DataLoader(  # 创建训练数据加载器
            datasets.MNIST(  # 加载MNIST数据集
                r"E:\desktop\Model-Compression-Demo-master\datasets",
                train=True, transform=self.trans, download=False),  # 设置为训练集并应用预处理，不下载
            batch_size=100, shuffle=True, drop_last=True  # 批次大小设为100，打乱顺序，丢弃最后一个不满批次的数据
        )
        self.optimizer = torch.optim.Adam(self.net.parameters())  # 使用Adam优化算法初始化优化器
        self.net.train()  # 将模型设置为训练模式

    def train(self):
        """
        训练模型。
        """
        for epoch in range(1, 3):  # 进行2个epoch的训练
            total = 0  # 初始化已处理样本数计数器
            for i, (data, label) in enumerate(self.train_data):  # 遍历数据加载器
                data, label = data.to(self.device), label.to(self.device)  # 将数据和标签移动到指定设备
                output = self.net(data)  # 前向传播，获取预测结果
                loss = self.net.get_loss(output, label)  # 计算损失
                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播，计算梯度
                self.optimizer.step()  # 更新参数

                total += len(data)  # 累加已处理的样本数
                progress = math.ceil(i / len(self.train_data) * 50)  # 计算进度条宽度
                print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')  # 打印进度信息
            torch.save(self.net.state_dict(), self.save_path)  # 每个epoch后保存模型权重
            # example = torch.Tensor(1, 1, 28, 28).to(self.device)  # 创建一个示例输入张量（被注释掉）
            # torch.jit.save(torch.jit.trace(self.net,example), self.save_path)  # 通过跟踪保存模型（被注释掉）


if __name__ == '__main__':
    trainer = Trainer("models/net.pth")  # 创建Trainer实例，指定模型保存路径
    trainer.train()  # 开始训练过程