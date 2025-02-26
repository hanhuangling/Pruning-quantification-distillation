import sys

# 将项目特定路径添加到系统路径中，以便能够导入自定义模块
sys.path.append(r'E:\desktop\Model-Compression-Demo-master')  # 添加项目特定路径到系统路径

from quantization.nets import MyNet  # 导入自定义网络模型MyNet
import torch  # 导入PyTorch库
from torchvision import datasets, transforms  # 导入数据集和预处理工具
from torch.utils.data import DataLoader  # 导入数据加载器
import time  # 导入时间库，用于计算推理时间
import os  # 导入操作系统库，用于文件操作


class Detector:
    def __init__(self, net_path, isQuantize=False):
        """
        初始化方法，设置设备、加载模型、数据集等。
        :param net_path: 模型权重或脚本化模型的路径。
        :param isQuantize: 是否加载的是量化后的模型。
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 设置运行设备为GPU或CPU
        # self.device = "cuda"
        self.net = MyNet().to(self.device)  # 实例化网络模型，并移动到指定设备上

        # 根据是否是量化模型来选择加载方式
        if isQuantize:
            self.net = torch.jit.load(net_path)
        else:
            self.net.load_state_dict(torch.load(net_path, map_location=self.device))

        # 定义图像转换，包括转换为张量和标准化
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 加载MNIST测试数据集，并创建DataLoader实例
        self.test_data = DataLoader(
            datasets.MNIST(r"E:\desktop\Model-Compression-Demo-master\datasets", train=False, transform=self.trans, download=False),
            batch_size=100, shuffle=True, drop_last=True)

        self.net.eval()  # 将模型设置为评估模式

    def detect(self):
        """执行模型检测（推理），并打印结果"""
        self.print_size_of_model()  # 打印模型大小
        test_loss = 0
        correct = 0
        start = time.time()

        with torch.no_grad():  # 禁用梯度计算以节省内存和加速推理
            for data, label in self.test_data:
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                pred = output.argmax(dim=1, keepdim=True)  # 获取预测标签
                correct += pred.eq(label.view_as(pred)).sum().item()  # 统计正确预测的数量

        end = time.time()
        print(f"total time:{end - start}")  # 打印总推理时间

        print('Test: accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(self.test_data.dataset),
                                                         100. * correct / len(self.test_data.dataset)))  # 打印准确率

    def quantize(self):
        """量化模型并进行评估"""
        print("=====quantize start=====")
        self.net.fuse_model()  # 融合模型中的某些操作以优化量化
        self.net.qconfig = torch.quantization.default_qconfig  # 设置默认量化配置
        torch.quantization.prepare(self.net, inplace=True)  # 准备量化模型

        # 转换为量化模型
        torch.quantization.convert(self.net, inplace=True)
        self.detect()  # 评估量化后的模型

    def print_size_of_model(self):
        """打印模型的大小（以MB为单位）"""
        torch.save(self.net.state_dict(), "temp.p")  # 临时保存模型状态字典
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')  # 删除临时文件


if __name__ == '__main__':
    # 创建不同模型的Detector实例，并调用detect方法评估它们
    detector_paths = [
        ("models/net.pth", False),
        # ("models/net_fuse.pth", True),
        ("models/net_convert.pth", True),
        ("models/net_convert_qat.pth", True)
    ]

    for path, is_quantized in detector_paths:
        print(path)
        detector = Detector(path, isQuantize=is_quantized)
        detector.detect()