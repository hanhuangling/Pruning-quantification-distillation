# 导入必要的PyTorch库和其他依赖项
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic  # 导入动态量化函数
import os  # 导入操作系统库，用于文件操作


class LeNet(nn.Module):
    def __init__(self):
        """
        初始化LeNet类，定义网络结构。
        """
        super(LeNet, self).__init__()  # 调用父类的初始化方法
        # 定义卷积层：输入1个通道，输出6个通道，使用3x3的卷积核
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 定义第二个卷积层：输入6个通道，输出16个通道，使用3x3的卷积核
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义第一个全连接层：输入为16*5*5（特征图大小），输出120个神经元
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义第二个全连接层：输入120个神经元，输出84个神经元
        self.fc2 = nn.Linear(120, 84)
        # 定义第三个全连接层（输出层）：输入84个神经元，输出10个类别
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        定义前向传播过程。
        :param x: 输入数据
        :return: 输出数据
        """
        # 第一层卷积后接ReLU激活和2x2的最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 第二层卷积后接ReLU激活和2x2的最大池化
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 展平多维张量，准备送入全连接层
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        # 通过第一层全连接层，并应用ReLU激活
        x = F.relu(self.fc1(x))
        # 通过第二层全连接层，并应用ReLU激活
        x = F.relu(self.fc2(x))
        # 通过第三层全连接层（输出层）
        x = self.fc3(x)
        return x


model = LeNet()  # 创建LeNet实例

# 使用quantize_dynamic对模型进行动态量化，指定要量化的模块类型和量化后的数据类型
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)  # 打印量化后的模型结构


def print_size_of_model(model):
    """
    计算并打印模型的大小（以MB为单位）。
    :param model: 要计算大小的模型
    """
    torch.save(model.state_dict(), "temp.p")  # 将模型的状态字典保存到临时文件中
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)  # 打印模型大小
    os.remove('temp.p')  # 删除临时文件


print_size_of_model(model)  # 打印原始模型的大小
print_size_of_model(quantized_model)  # 打印量化后模型的大小