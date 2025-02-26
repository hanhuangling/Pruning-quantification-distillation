# 导入必要的PyTorch库和其他依赖项
import torch
from torch import nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub  # 导入量化和反量化存根
import os  # 导入操作系统库，用于文件操作


class LeNet(nn.Module):
    def __init__(self):
        """
        初始化LeNet类，定义网络结构。
        """
        super(LeNet, self).__init__()  # 调用父类的初始化方法
        # 定义第一层卷积：输入1个通道，输出6个通道，使用3x3的卷积核
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)  # 定义第一层批归一化
        self.relu1 = nn.ReLU()  # 定义ReLU激活函数
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 定义2x2的最大池化

        # 定义第二层卷积：输入6个通道，输出16个通道，使用3x3的卷积核
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)  # 定义第二层批归一化
        self.relu2 = nn.ReLU()  # 定义ReLU激活函数
        self.maxpool2 = nn.MaxPool2d(2, 2)  # 定义2x2的最大池化

        # 定义第一个全连接层：输入为16*5*5（特征图大小），输出120个神经元
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()  # 定义ReLU激活函数

        # 定义第二个全连接层：输入120个神经元，输出84个神经元
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()  # 定义ReLU激活函数

        # 定义第三个全连接层（输出层）：输入84个神经元，输出10个类别
        self.fc3 = nn.Linear(84, 10)

        self.quant = QuantStub()  # 定义量化存根，用于模型输入的量化
        self.dequant = DeQuantStub()  # 定义反量化存根，用于模型输出的反量化

    def forward(self, x):
        """
        定义前向传播过程。
        :param x: 输入数据
        :return: 输出数据
        """
        # 对输入进行量化
        x = self.quant(x)
        # 第一层卷积、批归一化、ReLU激活和最大池化
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        # 第二层卷积、批归一化、ReLU激活和最大池化
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        # 展平多维张量，准备送入全连接层
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        # 通过第一层全连接层，并应用ReLU激活
        x = self.relu3(self.fc1(x))
        # 通过第二层全连接层，并应用ReLU激活
        x = self.relu4(self.fc2(x))
        # 通过第三层全连接层（输出层）
        x = self.fc3(x)
        # 对输出进行反量化
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """
        融合模型中的某些操作以优化量化。
        """
        # 融合Conv+BN+ReLU模块，提高量化效率
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                                               ['conv2', 'bn2', 'relu2'],
                                               ['fc1', 'relu3'], ['fc2', 'relu4']], inplace=True)


model = LeNet()  # 创建LeNet实例
print(model)  # 打印模型结构

model.eval()  # 将模型设置为评估模式，禁用dropout等训练时的行为

model.fuse_model()  # 调用fuse_model方法来融合模型中的模块，以便更好地进行量化

torch.save(model.state_dict(), "temp.p")  # 将模型的状态字典保存到临时文件中
print('Size (MB):', os.path.getsize("temp.p") / 1e6)  # 打印未量化模型大小
os.remove('temp.p')  # 删除临时文件

model.qconfig = torch.quantization.default_qconfig  # 设置默认量化配置
print(model.qconfig)  # 打印当前的量化配置（被注释掉）

torch.quantization.prepare(model, inplace=True)  # 准备模型进行静态量化

# Convert to quantized model
torch.quantization.convert(model, inplace=True)  # 将模型转换为量化模型
torch.save(model.state_dict(), "temp.p")  # 将量化后的模型状态字典保存到临时文件中
print('Size (MB):', os.path.getsize("temp.p") / 1e6)  # 打印量化后模型大小
os.remove('temp.p')  # 删除临时文件