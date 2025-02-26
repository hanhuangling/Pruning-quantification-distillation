import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub


class Conv(nn.Sequential):
    """定义一个基础的卷积块，包含Conv2d、BatchNorm2d和ReLU"""
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )


class ResBlock(nn.Module):
    """定义一个残差块，使用FloatFunctional来进行浮点数加法操作，以便后续量化"""
    def __init__(self, input_channels):
        super().__init__()
        self.layer = nn.Sequential(
            Conv(input_channels, input_channels // 2, 1),
            Conv(input_channels // 2, input_channels // 2, 3, 1, 1),
            Conv(input_channels // 2, input_channels, 1)
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, data):
        return self.skip_add.add(data, self.layer(data))  # 使用FloatFunctional进行跳跃连接的加法操作


class MyNet(nn.Module):
    """定义主网络结构"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            Conv(1, 32, 3, 1, 1),
            ResBlock(32),
            nn.MaxPool2d(2),
            Conv(32, 64, 3, 1, 1),
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2),
            Conv(64, 64, 3, 1, 1),
        )

        self.linear1 = nn.Linear(7 * 7 * 64, 128)  # 假设输入图像尺寸为28x28，经过两次MaxPool后变为7x7
        self.relu4 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(128, 10)  # 对于MNIST数据集，输出层有10个类别
        self.quant = QuantStub()  # 量化存根，用于模型输入的量化
        self.dequant = DeQuantStub()  # 反量化存根，用于模型输出的反量化
        self.loss = nn.CrossEntropyLoss()  # 定义损失函数

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        data = self.quant(data)  # 输入量化
        out = self.layer(data)
        out = out.reshape(out.size(0), -1)  # 展平特征图以适应全连接层输入
        out = self.relu4(self.linear1(out))
        out = self.linear2(out)
        out = self.dequant(out)  # 输出反量化
        return out

    def get_loss(self, output, label):
        return self.loss(output, label)  # 计算并返回损失值

    def fuse_model(self):
        """融合模型中的某些操作以优化量化"""
        for m in self.modules():
            if type(m) == Conv:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)  # 融合Conv-BN-Relu模块
        torch.quantization.fuse_modules(self, ['linear1', 'relu4'], inplace=True)  # 融合线性层和ReLU


if __name__ == '__main__':
    net = MyNet()
    net.fuse_model()  # 调用fuse_model方法来融合模型中的模块，以便更好地进行量化