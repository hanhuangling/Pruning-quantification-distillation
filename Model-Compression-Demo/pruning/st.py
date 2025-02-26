import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# 设备选择，优先使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    """定义LeNet模型类"""

    def __init__(self):
        super(LeNet, self).__init__()
        # 定义卷积层和全连接层
        self.conv1 = nn.Conv2d(1, 6, 3)  # 第一层卷积层，输入通道数为1，输出通道数为6，卷积核大小为3x3
        self.conv2 = nn.Conv2d(6, 16, 3)  # 第二层卷积层，输入通道数为6，输出通道数为16，卷积核大小为3x3
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 第一层全连接层，输入特征数为16*5*5，输出特征数为120
        self.fc2 = nn.Linear(120, 84)  # 第二层全连接层，输入特征数为120，输出特征数为84
        self.fc3 = nn.Linear(84, 10)  # 输出层，输入特征数为84，输出特征数为10（类别数）
        # self.dropout = nn.Dropout(0.8)  # Dropout层，未在forward中使用

    def forward(self, x):
        """定义前向传播过程"""
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 卷积+ReLU激活+最大池化
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 再次卷积+ReLU激活+最大池化
        x = x.view(-1, int(x.nelement() / x.shape[0]))  # 将多维张量展平成二维张量
        x = F.relu(self.fc1(x))  # 全连接层+ReLU激活
        # x = self.dropout(self.fc1(x))  # Dropout层，未启用
        x = F.relu(self.fc2(x))  # 再次全连接层+ReLU激活
        x = self.fc3(x)  # 输出层
        return x


# 实例化LeNet模型，并移动到指定设备上
model = LeNet().to(device=device)

# 指定要剪枝的模块（这里选择了conv1作为示例）
module = model.conv1

# 打印出模块中的参数名称及其对应的参数值
print(list(module.named_parameters()))
for name, param in module.named_parameters():
    print('--------------------------------------')
    print(f"Parameter Name: {name}, Shape: {param.shape}")
    print('--------------------------------------')
# 打印出模块中的缓冲区名称及其对应的缓冲区值
print(list(module.named_buffers()))

# 对conv1的权重应用随机非结构化剪枝，剪枝比例为30%
prune.random_unstructured(module, name="weight", amount=0.3)

# 打印剪枝后的权重矩阵
print(module.weight)

# 对conv1的偏置应用基于L1范数的非结构化剪枝，剪枝掉最小的3个元素
prune.l1_unstructured(module, name="bias", amount=3)

# 打印剪枝后的偏置向量
print(module.bias)

# 对conv1的权重应用基于L2范数的结构化剪枝，沿着dim=0维度剪枝掉一半的通道
prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)

# 打印剪枝后的权重矩阵，验证是否正确地剪枝了50%的通道
print(module.weight)
print(module.weight.shape)

# 遍历_forward_pre_hooks字典，找到与权重相关的剪枝钩子并打印其历史记录
for hook in module._forward_pre_hooks.values():
    if hook._tensor_name == "weight":  # 选择出正确的钩子
        break

# 打印剪枝历史记录
print(list(hook))

# 打印模型状态字典的所有键名，以查看所有层的状态
print(model.state_dict().keys())

# 打印剪枝后模块的参数名称及其对应的参数值
print(list(module.named_parameters()))

# 移除conv1上的权重和偏置剪枝
prune.remove(module, 'weight')
prune.remove(module, 'bias')

# 打印移除剪枝后的模块参数列表
print(list(module.named_parameters()))

# 创建一个新的LeNet实例，并对所有卷积层和线性层进行剪枝
new_model = LeNet()
for name, module in new_model.named_modules():
    # 如果是2D卷积层，则对其权重应用20%的非结构化L1剪枝
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
    # 如果是线性层，则对其权重应用40%的非结构化L1剪枝
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)

# 打印新的模型中所有掩码的键名，以验证所有掩码是否存在
print(dict(new_model.named_buffers()).keys())

# 创建一个LeNet实例，并对其进行全局非结构化剪枝
model = LeNet()

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

# 对指定的参数应用全局非结构化剪枝，剪枝比例为20%
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

# 打印每个层的稀疏性信息，即有多少权重被置零
print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    )
)
print(
    "Sparsity in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv2.weight == 0))
        / float(model.conv2.weight.nelement())
    )
)
print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc1.weight == 0))
        / float(model.fc1.weight.nelement())
    )
)
print(
    "Sparsity in fc2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc2.weight == 0))
        / float(model.fc2.weight.nelement())
    )
)
print(
    "Sparsity in fc3.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc3.weight == 0))
        / float(model.fc3.weight.nelement())
    )
)
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
            + torch.sum(model.fc3.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
            + model.fc3.weight.nelement()
        )
    )
)


class FooBarPruningMethod(prune.BasePruningMethod):
    """自定义剪枝方法，用于剪枝张量中的每第二个元素"""

    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0  # 将张量展平后，每隔一个元素设置为0
        return mask


def foobar_unstructured(module, name):
    """使用自定义的FooBarPruningMethod对`module`中的名为`name`的参数执行剪枝

    Args:
        module (nn.Module): 包含要剪枝的张量的模块
        name (string): 模块中要剪枝的参数名
    """
    FooBarPruningMethod.apply(module, name)
    return module


# 使用自定义的foobar_unstructured函数对fc3层的偏置执行剪枝
model = LeNet()
foobar_unstructured(model.fc3, name='bias')

# 打印fc3层的偏置掩码，以验证剪枝效果
print(model.fc3.bias_mask)