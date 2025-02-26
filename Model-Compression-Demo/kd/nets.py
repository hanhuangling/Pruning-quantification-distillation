# 导入必要的PyTorch库和模块
import torch
from torch import nn
import torch.nn.functional as F

# 定义教师网络类TeacherNet，继承自nn.Module
class TeacherNet(nn.Module):
    def __init__(self):
        # 调用父类的构造函数进行初始化
        super(TeacherNet, self).__init__()
        # 定义第一个卷积层，输入通道为1（灰度图像），输出32个通道，卷积核大小为3x3，步长为1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 定义第二个卷积层，输入通道为32，输出64个通道，卷积核大小为3x3，步长为1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # 定义第一个Dropout层，用于防止过拟合，随机丢弃30%的特征图
        self.dropout1 = nn.Dropout2d(0.3)
        # 定义第二个Dropout层，随机丢弃50%的特征图
        self.dropout2 = nn.Dropout2d(0.5)
        # 定义第一个全连接层，输入特征数为9216（即64*12*12），输出128个特征
        self.fc1 = nn.Linear(9216, 128)
        # 定义第二个全连接层，输入128个特征，输出10个分类结果
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 应用第一个卷积层到输入张量x，并激活ReLU非线性变换
        x = self.conv1(x)
        x = F.relu(x)
        # 应用第二个卷积层并再次激活ReLU
        x = self.conv2(x)
        x = F.relu(x)
        # 应用最大池化层，窗口大小为2x2，减少特征图尺寸
        x = F.max_pool2d(x, 2)
        # 应用第一个Dropout层，减少过拟合风险
        x = self.dropout1(x)
        # 将多维张量展平成一维向量，准备送入全连接层
        x = torch.flatten(x, 1)
        # 应用第一个全连接层并激活ReLU
        x = self.fc1(x)
        x = F.relu(x)
        # 应用第二个Dropout层
        x = self.dropout2(x)
        # 应用第二个全连接层得到最终输出
        output = self.fc2(x)
        return output

# 定义学生网络类StudentNet，也继承自nn.Module
class StudentNet(nn.Module):
    def __init__(self):
        # 同样调用父类的构造函数进行初始化
        super(StudentNet, self).__init__()
        # 定义第一个全连接层，输入为28x28（假设是MNIST数据集的输入大小），输出128个特征
        self.fc1 = nn.Linear(28 * 28, 128)
        # 定义第二个全连接层，输入128个特征，输出64个特征
        self.fc2 = nn.Linear(128, 64)
        # 定义第三个全连接层，输入64个特征，输出10个分类结果
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 展平输入张量x，使其变为一维向量
        x = torch.flatten(x, 1)
        # 应用第一个全连接层并激活ReLU
        x = F.relu(self.fc1(x))
        # 应用第二个全连接层并再次激活ReLU
        x = F.relu(self.fc2(x))
        # 应用第三个全连接层，注意这里原代码中使用了ReLU激活，通常在最后一层不使用ReLU以保留原始输出值
        output = F.relu(self.fc3(x))  # 建议移除这里的ReLU激活函数
        return output