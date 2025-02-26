import torch  # 导入PyTorch库
import sys  # 导入sys库用于系统路径操作

# 将项目特定路径添加到系统路径中，以便能够导入自定义模块
sys.path.append(r'E:\desktop\Model-Compression-Demo-master')

# 从自定义模块pruning.nets中导入MyNet类，即要进行剪枝的网络模型
from pruning.nets import MyNet
# 从自定义模块pruning.utils中导入weight_prune和filter_prune函数，用于实现不同的剪枝策略
from pruning.utils import weight_prune, filter_prune
# 从torch.nn.utils.prune中导入prune模块，提供官方剪枝工具
import torch.nn.utils.prune as prune


class Pruning:
    """定义一个剪枝类，负责加载预训练模型并执行剪枝操作"""

    def __init__(self, net_path, amount):
        """初始化方法，加载预训练模型并指定要剪枝的比例"""
        self.net = MyNet()  # 实例化要剪枝的网络模型
        self.net.load_state_dict(torch.load(net_path))  # 加载预训练的权重参数
        # 定义要剪枝的层及其参数名（这里是所有卷积层和全连接层的权重）
        self.parameters_to_prune = (
            (self.net.conv1, 'weight'),
            (self.net.conv2, 'weight'),
            (self.net.conv3, 'weight'),
            (self.net.linear1, 'weight'),
            (self.net.linear2, 'weight'),
        )
        self.amount = amount  # 设置剪枝比例

    def pruning(self):
        """执行全局非结构化剪枝"""
        prune.global_unstructured(
            self.parameters_to_prune,  # 指定要剪枝的参数
            pruning_method=prune.L1Unstructured,  # 使用L1范数作为剪枝标准
            amount=self.amount,  # 剪枝比例
        )

        # 移除由PyTorch剪枝API自动添加的mask和原始权重参数，恢复为单一的、剪枝后的权重张量
        for module in [self.net.conv1, self.net.conv2, self.net.conv3, self.net.linear1, self.net.linear2]:
            prune.remove(module, 'weight')

        # 打印每个层的稀疏性信息（即有多少权重被置零）
        for layer_name, layer in [('conv1', self.net.conv1), ('conv2', self.net.conv2),
                                  ('conv3', self.net.conv3), ('linear1', self.net.linear1),
                                  ('linear2', self.net.linear2)]:
            sparsity = 100. * float(torch.sum(layer.weight == 0)) / float(layer.weight.nelement())
            print(f"Sparsity in {layer_name}.weight: {sparsity:.2f}%")

        # 计算并打印全局稀疏性
        total_zeros = sum([torch.sum(layer.weight == 0) for layer in [self.net.conv1, self.net.conv2,
                                                                      self.net.conv3, self.net.linear1,
                                                                      self.net.linear2]])
        total_elements = sum([layer.weight.nelement() for layer in [self.net.conv1, self.net.conv2,
                                                                    self.net.conv3, self.net.linear1,
                                                                    self.net.linear2]])
        global_sparsity = 100. * float(total_zeros) / float(total_elements)
        print(f"Global sparsity: {global_sparsity:.2f}%")

        # 保存剪枝后的模型权重
        torch.save(self.net.state_dict(), f"models/pruned_net_with_torch_{self.amount:.1f}_l1.pth")


if __name__ == '__main__':
    # 对一系列不同剪枝比例进行剪枝并保存模型
    for i in range(1, 10):
        amount = 0.1 * i  # 计算当前剪枝比例
        print(f"Pruning with amount: {amount}")  # 打印当前剪枝比例
        pruning = Pruning("models/net.pth", amount)  # 创建剪枝实例，加载原始模型
        pruning.pruning()  # 执行剪枝操作