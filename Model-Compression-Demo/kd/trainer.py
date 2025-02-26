# 导入PyTorch库及其相关模块，包括数据集处理、转换和数据加载器
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math  # 导入math库用于数学运算
import sys  # 导入sys库用于系统路径操作

sys.path.append(r'E:\desktop\Model-Compression-Demo-master')  # 添加项目特定路径到系统路径

# 从自定义模块kd.nets中导入教师网络和学生网络的定义
from kd.nets import TeacherNet, StudentNet
from torch import nn  # 导入神经网络相关的模块
from torch.nn import functional as F  # 导入函数式API（如激活函数等）


# 定义训练器类Trainer，用于管理教师网络和学生网络的训练过程
class Trainer:
    def __init__(self):
        # 检测是否有可用的CUDA设备，否则使用CPU进行计算
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 初始化教师网络，并将其移动到指定的计算设备上
        self.teacher_net = TeacherNet().to(self.device)
        # 初始化学生网络，并将其移动到指定的计算设备上
        self.student_net = StudentNet().to(self.device)

        # 定义图像预处理转换，包括转换为张量和标准化
        self.trans = transforms.Compose([
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor
            transforms.Normalize([0.1307], [0.3081])  # 对图像数据进行标准化
        ])

        # 加载MNIST训练数据集，应用上述转换，并创建DataLoader
        self.train_data = DataLoader(
            datasets.MNIST(r"E:\desktop\Model-Compression-Demo-master\datasets", train=True,
                           transform=self.trans, download=False),
            batch_size=1000, shuffle=True, drop_last=True)  # 设置batch大小、是否打乱数据以及是否丢弃最后一个不完整的batch

        # 加载MNIST测试数据集，创建DataLoader
        self.test_data = DataLoader(
            datasets.MNIST(r"E:\desktop\Model-Compression-Demo-master\datasets",
                           train=False, transform=self.trans, download=False),
            batch_size=10000, shuffle=True)  # 设置batch大小和是否打乱数据

        # 定义交叉熵损失函数，用于分类任务
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        # 定义KL散度损失函数，用于知识蒸馏
        self.KLDivLoss = nn.KLDivLoss()

        # 定义教师网络的优化器，采用Adam算法
        self.teacher_optimizer = torch.optim.Adam(self.teacher_net.parameters())
        # 定义学生网络的优化器，同样采用Adam算法
        self.student_optimizer = torch.optim.Adam(self.student_net.parameters())

    def train_teacher(self):
        # 将教师网络设置为训练模式
        self.teacher_net.train()

        # 训练教师网络两个epoch
        for epoch in range(1, 3):
            total = 0  # 初始化计数器以追踪已处理的数据量
            # 遍历训练数据集中的每一个batch
            for i, (data, label) in enumerate(self.train_data):
                # 将数据和标签移动到指定的计算设备上
                data, label = data.to(self.device), label.to(self.device)

                # 前向传播：将数据输入教师网络并获取输出
                output = self.teacher_net(data)

                # 计算损失值
                loss = self.CrossEntropyLoss(output, label)

                # 反向传播和参数更新
                self.teacher_optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                self.teacher_optimizer.step()  # 更新参数

                # 更新已处理的数据总量
                total += len(data)

                # 打印进度条显示训练进度
                progress = math.ceil(i / len(self.train_data) * 50)
                print("\rTrain teacher_net epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')

            # 在每个epoch结束时评估教师网络性能（取消了保存模型的步骤）
            torch.save(self.teacher_net.state_dict(),r"E:\desktop\Model-Compression-Demo-master\kd\models\teacher_net.pth")
            self.evaluate(self.teacher_net)

    def train_student(self):
        # 将学生网络设置为训练模式
        self.student_net.train()

        # 训练学生网络两个epoch
        for epoch in range(1, 5):
            total = 0  # 初始化计数器以追踪已处理的数据量
            # 遍历训练数据集中的每一个batch
            for i, (data, label) in enumerate(self.train_data):
                # 将数据和标签移动到指定的计算设备上
                data, label = data.to(self.device), label.to(self.device)

                # 获取教师网络和学生网络的输出
                with torch.no_grad():  # 禁用梯度计算以提高效率
                    teacher_output = self.teacher_net(data)
                student_output = self.student_net(data)

                # 分离教师网络输出，防止其参与反向传播
                teacher_output = teacher_output.detach()

                # 计算损失值，结合交叉熵损失和蒸馏损失
                loss = self.distillation(student_output, label, teacher_output, temp=5.0, alpha=0.7)

                # 反向传播和参数更新
                self.student_optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                self.student_optimizer.step()  # 更新参数

                # 更新已处理的数据总量
                total += len(data)

                # 打印进度条显示训练进度
                progress = math.ceil(i / len(self.train_data) * 50)
                print("\rTrain student_net epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')

            # 在每个epoch结束时评估学生网络性能（取消了保存模型的步骤）
            torch.save(self.student_net.state_dict(),
                       r"E:\desktop\Model-Compression-Demo-master\kd\models\student_net.pth")
            self.evaluate(self.student_net)

    def evaluate(self, net):
        # 将网络设置为评估模式
        net.eval()

        # 使用测试数据集评估网络性能
        for data, label in self.test_data:
            # 将数据和标签移动到指定的计算设备上
            data, label = data.to(self.device), label.to(self.device)

            # 前向传播：将数据输入网络并获取输出
            output = net(data)

            # 获取预测结果的最大概率对应的类别索引
            pred = output.argmax(dim=1, keepdim=True)

            # 计算准确率
            acc = pred.eq(label.view_as(pred)).sum().item() / self.test_data.batch_size

            # 打印评估结果
            print(f"\nevaluate acc:{acc * 100:.2f}%")

    # 定义知识蒸馏方法，结合教师网络和学生网络的输出
    def distillation(self, y, labels, teacher_scores, temp, alpha):
        # 蒸馏损失由两部分组成：基于教师网络输出的软目标损失和基于真实标签的硬目标损失
        return self.KLDivLoss(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
                temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


# 如果脚本作为主程序运行，则执行以下代码
if __name__ == '__main__':
    # 创建训练器实例
    trainer = Trainer()

    # 开始训练教师网络
    trainer.train_teacher()

    # 开始训练学生网络
    trainer.train_student()