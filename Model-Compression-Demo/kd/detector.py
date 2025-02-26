# 导入必要的PyTorch库及其相关模块，包括数据集处理、转换和数据加载器
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math  # 导入math库用于数学运算
import sys  # 导入sys库用于系统路径操作

sys.path.append(r'E:\desktop\Model-Compression-Demo-master')  # 添加项目特定路径到系统路径

# 从自定义模块kd.nets中导入教师网络和学生网络的定义
from kd.nets import TeacherNet, StudentNet
from torch import nn  # 导入神经网络相关的模块
import time  # 导入time库用于计算时间消耗


# 定义检测器类Detector，用于测试教师网络或学生网络的性能
class Detector:
    def __init__(self, net_path, isTeacher=True):
        # 检测是否有可用的CUDA设备，否则使用CPU进行计算
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 根据isTeacher参数选择初始化教师网络或学生网络，并将其移动到指定的计算设备上
        self.net = TeacherNet().to(self.device) if isTeacher else StudentNet().to(self.device)

        # 定义图像预处理转换，包括转换为张量和标准化
        self.trans = transforms.Compose([
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor
            transforms.Normalize([0.1307], [0.3081])  # 对图像数据进行标准化
        ])

        # 加载MNIST测试数据集，应用上述转换，并创建DataLoader
        self.test_data = DataLoader(
            datasets.MNIST(r"E:\点头第五期课程\模型剪枝和量化和蒸馏\Model-Compression-Demo-master\datasets",
                           train=False, transform=self.trans, download=False),
            batch_size=100, shuffle=True)  # 设置batch大小和是否打乱数据

        # 加载预训练的网络权重文件，并将它们映射到当前使用的计算设备上
        self.net.load_state_dict(torch.load(net_path, map_location=self.device))

        # 将网络设置为评估模式，禁用dropout等只在训练时启用的操作
        self.net.eval()

    def detect(self):
        correct = 0  # 初始化计数器以追踪正确预测的数量
        start = time.time()  # 记录开始时间

        # 禁用梯度计算以提高效率
        with torch.no_grad():
            # 遍历测试数据集中的每一个batch
            for data, label in self.test_data:
                # 将数据和标签移动到指定的计算设备上
                data, label = data.to(self.device), label.to(self.device)

                # 前向传播：将数据输入网络并获取输出
                output = self.net(data)

                # 获取预测结果的最大概率对应的类别索引
                pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability

                # 统计正确预测的数量
                correct += pred.eq(label.view_as(pred)).sum().item()

        end = time.time()  # 记录结束时间

        # 打印总耗时
        print(f"total time:{end - start}")

        # 打印测试结果，包括正确预测数量、总样本数量以及准确率百分比
        print('Test: average  accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(self.test_data.dataset),
                                                                  100. * correct / len(self.test_data.dataset)))


# 如果脚本作为主程序运行，则执行以下代码
if __name__ == '__main__':
    # 测试教师网络性能
    print("teacher_net")
    detector = Detector("models/teacher_net.pth")  # 创建检测器实例，加载教师网络权重
    detector.detect()  # 开始测试教师网络

    # 测试学生网络性能
    print("student_net")
    detector = Detector("models/student_net.pth", False)  # 创建检测器实例，加载学生网络权重
    detector.detect()  # 开始测试学生网络