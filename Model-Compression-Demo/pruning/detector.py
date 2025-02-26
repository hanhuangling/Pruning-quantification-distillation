# 导入必要的系统路径操作库
import sys

sys.path.append(r'E:\desktop\Model-Compression-Demo-master')  # 添加项目特定路径到系统路径

# 从自定义模块中导入所需的网络模型、工具函数等
from pruning.nets import MyNet  # 自定义网络模型
import torch  # PyTorch库
from copy import deepcopy  # 深拷贝函数用于复制对象
from torchvision import datasets, transforms  # 数据集处理和转换
from torch.utils.data import DataLoader  # 数据加载器
from pruning.utils import weight_prune, plot_weights, filter_prune  # 剪枝相关工具函数
import time  # 时间测量库


class Detector:
    def __init__(self, net_path):
        # 根据是否有可用的CUDA设备选择计算设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化MyNet网络，并将其移动到指定的计算设备上
        self.net = MyNet().to(self.device)

        # 定义图像预处理转换，包括转换为张量和标准化（这里使用0.5作为均值和标准差）
        self.trans = transforms.Compose([
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor
            transforms.Normalize([0.5], [0.5])  # 对图像数据进行标准化
        ])

        # 加载MNIST测试数据集，应用上述转换，并创建DataLoader
        self.test_data = DataLoader(
            datasets.MNIST("../datasets/", train=False, transform=self.trans, download=False),
            batch_size=100, shuffle=True, drop_last=True)  # 设置batch大小、是否打乱数据以及是否丢弃最后一个不完整的batch

        # 根据是否有GPU来确定如何加载模型参数
        self.map_location = None if torch.cuda.is_available() else lambda storage, loc: storage

        # 加载预训练的网络权重文件，并将它们映射到当前使用的计算设备上
        self.net.load_state_dict(torch.load(net_path, map_location=self.map_location))

        # 将网络设置为评估模式，禁用dropout等只在训练时启用的操作
        self.net.eval()

    def detect(self):
        test_loss = 0  # 初始化测试损失计数器
        correct = 0  # 初始化正确预测计数器
        start = time.time()  # 记录开始时间

        # 禁用梯度计算以提高效率
        with torch.no_grad():
            # 遍历测试数据集中的每一个batch
            for data, label in self.test_data:
                # 将数据和标签移动到指定的计算设备上
                data, label = data.to(self.device), label.to(self.device)

                # 前向传播：将数据输入网络并获取输出
                output = self.net(data)

                # 累加测试损失
                test_loss += self.net.get_loss(output, label).item()

                # 获取预测结果的最大概率对应的类别索引
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                # 统计正确预测的数量
                correct += pred.eq(label.view_as(pred)).sum().item()

        end = time.time()  # 记录结束时间

        # 打印总耗时
        print(f"total time:{end - start}")

        # 平均测试损失
        test_loss /= len(self.test_data.dataset)

        # 打印测试结果，包括平均损失、正确预测数量、总样本数量以及准确率百分比
        print('Test: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_data.dataset),
            100. * correct / len(self.test_data.dataset)))


if __name__ == '__main__':
    # 测试原始网络性能
    print("models/net.pth")
    detector1 = Detector("models/net.pth")  # 创建检测器实例，加载原始网络权重
    detector1.detect()  # 开始测试原始网络

    # 测试一系列不同剪枝比例的网络性能
    for i in range(1, 10):
        amount = 0.1 * i  # 计算剪枝比例
        # 测试L1范数剪枝后的网络性能
        print(f"models/pruned_net_with_torch_{amount:.1f}_l1.pth")
        detector1 = Detector(f"models/pruned_net_with_torch_{amount:.1f}_l1.pth")
        detector1.detect()

