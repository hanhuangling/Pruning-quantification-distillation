import torch
import sys

# 将项目特定路径添加到系统路径中，以便能够导入自定义模块
sys.path.append(r'E:\desktop\Model-Compression-Demo-master')  # 添加项目特定路径到系统路径
from quantization.nets import MyNet  # 从指定路径导入自定义网络模型MyNet
import os  # 导入操作系统库，用于文件操作


class Quantize:
    def __init__(self, net_path):
        """
        初始化Quantize类，设置设备、加载并融合模型。
        :param net_path: 模型权重的路径。
        """
        self.device = "cpu" if torch.cuda.is_available() else "cpu"  # 设置运行设备为GPU或CPU
        self.net = MyNet().to(self.device)  # 实例化网络模型，并移动到指定设备上
        self.net.load_state_dict(torch.load(net_path, map_location=self.device))  # 加载预训练模型权重
        self.net.eval()  # 将模型设置为评估模式，以禁用dropout等训练时的行为
        self.net.fuse_model()  # 融合模型中的某些操作以优化量化

    def quantize(self):
        """
        对模型进行量化，并保存量化前后的模型。
        """
        # example = torch.Tensor(1, 1, 28, 28).to(self.device)  # 创建一个示例输入张量（被注释掉）
        # torch.jit.save(torch.jit.trace(self.net, example), "models/net_fuse.pth")  # 通过跟踪保存融合后的模型（被注释掉）

        # 使用TorchScript直接保存融合后的模型
        torch.jit.save(torch.jit.script(self.net), "models/net_fuse.pth")

        # 打印融合后模型的大小（以MB为单位）
        print('Size (MB):', os.path.getsize("models/net_fuse.pth") / 1e6)

        self.net.qconfig = torch.quantization.default_qconfig  # 设置默认量化配置

        # 打印当前的量化配置（被注释掉）
        # print(model.qconfig)

        # 注意：量化只适用于CPU设备
        torch.quantization.prepare(self.net, inplace=True)  # 准备模型进行量化

        # 将模型转换为量化模型
        torch.quantization.convert(self.net, inplace=True)

        # 使用TorchScript保存量化后的模型
        torch.jit.save(torch.jit.script(self.net), "models/net_convert.pth")

        # 打印量化后模型的大小（以MB为单位）
        print('Size (MB):', os.path.getsize("models/net_convert.pth") / 1e6)


if __name__ == '__main__':
    quantize = Quantize("models/net.pth")  # 创建Quantize实例，加载原始模型
    quantize.quantize()  # 执行量化过程