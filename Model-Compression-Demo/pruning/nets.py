# @desc :编写网络以及剪枝方面的调用函数

import torch
from torch import nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入函数式API（如激活函数等）
import sys  # 导入sys库用于系统路径操作

# 将项目特定路径添加到系统路径中，以便能够导入自定义模块
sys.path.append(r'E:\desktop\Model-Compression-Demo-master')  # 添加项目特定路径到系统路径

# 从自定义模块pruning.utils中导入to_var函数，用于将numpy数组转换为torch变量
from pruning.utils import to_var


class MaskedConv2d(nn.Conv2d):
    """继承自nn.Conv2d的卷积层，添加了mask机制以支持剪枝"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        # 调用父类初始化方法
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False  # 标记是否应用了mask

    def set_mask(self, mask):
        """设置当前层的mask，并更新权重"""
        self.mask = to_var(mask, requires_grad=False)  # 将mask转换为torch变量且不计算梯度
        self.weight.data = self.weight.data * self.mask.data  # 应用mask到权重上
        self.mask_flag = True  # 更新mask标记为已应用

    def get_mask(self):
        """获取当前层的mask状态"""
        print(self.mask_flag)  # 打印mask是否已应用
        return self.mask  # 返回mask

    def forward(self, data):
        """重写forward方法，根据mask_flag决定是否应用mask"""
        if self.mask_flag:
            weight = self.weight * self.mask  # 如果有mask，则使用mask后的权重
        else:
            weight = self.weight  # 否则使用原始权重
        # 执行卷积操作并返回结果
        return F.conv2d(data, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MaskedLinear(nn.Linear):
    """继承自nn.Linear的全连接层，添加了mask机制以支持剪枝"""

    def __init__(self, in_channels, out_channels, bias=True):
        # 调用父类初始化方法
        super().__init__(in_channels, out_channels, bias)
        self.mask_flag = False  # 标记是否应用了mask

    def set_mask(self, mask):
        """设置当前层的mask，并更新权重"""
        self.mask = to_var(mask, requires_grad=False)  # 将mask转换为torch变量且不计算梯度
        self.weight.data = self.weight.data * self.mask.data  # 应用mask到权重上
        self.mask_flag = True  # 更新mask标记为已应用

    def get_mask(self):
        """获取当前层的mask状态"""
        print(self.mask_flag)  # 打印mask是否已应用
        return self.mask  # 返回mask

    def forward(self, data):
        """重写forward方法，根据mask_flag决定是否应用mask"""
        if self.mask_flag:
            weight = self.weight * self.mask  # 如果有mask，则使用mask后的权重
        else:
            weight = self.weight  # 否则使用原始权重
        # 执行线性变换并返回结果
        return F.linear(data, weight, self.bias)


class MyNet(nn.Module):
    """自定义神经网络模型，包含MaskedConv2d和MaskedLinear层以支持剪枝"""

    def __init__(self):
        super().__init__()
        # 定义卷积层、激活函数和池化层
        self.conv1 = MaskedConv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = MaskedConv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = MaskedConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        # 定义全连接层
        self.linear1 = MaskedLinear(7 * 7 * 64, 128)
        self.linear2 = MaskedLinear(128, 10)
        # self.drop=nn.Dropout(0.1)
        # 定义损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, data):
        """定义前向传播过程"""
        out = self.maxpool1(self.relu1(self.conv1(data)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)  # 展平输出
        out = self.linear1(out)
        # out=self.drop(out)
        out = self.linear2(out)
        return out

    def get_loss(self, output, label):
        """计算交叉熵损失"""
        return self.loss(output, label)

    def set_masks(self, masks, isLinear=False):
        """为网络中的层设置mask"""
        if isLinear:
            self.linear1.set_mask(masks[0])
            self.linear2.set_mask(masks[1])
        else:
            self.conv1.set_mask(torch.from_numpy(masks[0]))
            self.conv2.set_mask(torch.from_numpy(masks[1]))
            self.conv3.set_mask(torch.from_numpy(masks[2]))


if __name__ == '__main__':
    net = MyNet()
    for p in net.conv1.parameters():
        print(p.data.size())  # 打印第一个卷积层参数的尺寸
    for p in net.linear1.parameters():
        print(p.data.size())  # 打印第一个全连接层参数的尺寸