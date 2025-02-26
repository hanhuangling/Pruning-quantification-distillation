import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库用于数值计算
from matplotlib import pyplot as plt  # 导入Matplotlib库用于绘图


def to_var(x, requires_grad=False):
    """
    Automatically choose cpu or cuda and return a new tensor with the same data but on the selected device.
    """
    if torch.cuda.is_available():  # 检查是否有可用的CUDA设备（GPU）
        x = x.cuda()  # 如果有CUDA设备，则将张量移动到GPU上
    return x.clone().detach().requires_grad_(requires_grad)  # 返回一个新的张量，指定是否需要梯度


def weight_prune(model, pruning_perc):
    '''
    Prune 'pruning_perc' % of weights layer-wise based on their absolute value.
    '''
    threshold_list = []  # 存储每层的剪枝阈值
    for p in model.parameters():  # 遍历模型的所有参数
        if len(p.data.size()) != 1:  # 排除偏置项（bias），只处理权重
            if len(p.data.size()) != 4:  # 这里可能是一个逻辑错误，通常应包含卷积层
                # 计算当前层权重的绝对值，并将其展平为一维数组
                weight = p.cpu().data.abs().numpy().flatten()
                # 根据剪枝百分比计算阈值，并存储到列表中
                threshold = np.percentile(weight, pruning_perc)
                threshold_list.append(threshold)

    # generate mask
    masks = []  # 存储每个层的掩码
    idx = 0  # 索引变量，用于遍历threshold_list
    for p in model.parameters():  # 再次遍历所有参数，这次是生成掩码
        if len(p.data.size()) != 1:
            if len(p.data.size()) != 4:  # 同样的逻辑错误
                # 创建一个布尔掩码，表示哪些权重应该保留（大于阈值）
                pruned_inds = p.data.abs() > threshold_list[idx]
                # 将布尔掩码转换为浮点数格式，并添加到masks列表中
                masks.append(pruned_inds.float())
                idx += 1  # 更新索引
    return masks  # 返回生成的掩码列表


"""Reference https://github.com/zepx/pytorch-weight-prune/"""


def prune_rate(model, verbose=False):
    """
    Calculate and print out the prune rate for each layer and the whole network.
    """
    total_nb_param = 0  # 统计整个网络的参数总数
    nb_zero_param = 0  # 统计被剪枝掉的参数数量

    layer_id = 0  # 层ID，用于打印信息时标识不同层

    for parameter in model.parameters():  # 遍历模型的所有参数

        param_this_layer = 1  # 当前层的参数数量
        for dim in parameter.data.size():  # 计算当前层的参数数量
            param_this_layer *= dim
        total_nb_param += param_this_layer  # 累加到总参数数量

        # 只对卷积层和全连接层进行剪枝率计算
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = np.count_nonzero(parameter.cpu().data.numpy() == 0)  # 统计当前层零值参数的数量
            nb_zero_param += zero_param_this_layer  # 累加到总的零值参数数量

            if verbose:  # 如果verbose为True，则打印详细的剪枝信息
                print("Layer {} | {} layer | {:.2f}% parameters pruned".format(
                    layer_id,
                    'Conv' if len(parameter.data.size()) == 4 else 'Linear',
                    100. * zero_param_this_layer / param_this_layer,
                ))
    pruning_perc = 100. * nb_zero_param / total_nb_param  # 计算全局剪枝率
    if verbose:  # 打印最终的剪枝率
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc  # 返回剪枝率


def arg_nonzero_min(a):
    """
    Find the index of the smallest non-zero element in a non-negative array.
    """

    if not a:  # 如果输入数组为空，则直接返回
        return

    min_ix, min_v = None, None  # 初始化最小值及其索引
    # 找到第一个非零元素作为初始值
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
            break
    if not min_ix:  # 如果没有找到非零元素，则打印警告并返回无穷大
        print('Warning: all zero')
        return np.inf, np.inf

    # 在剩余的元素中寻找更小的非零值
    for i, e in enumerate(a):
        if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix  # 返回最小值及其索引


def prune_one_filter(model, masks):
    '''
    Prune one least important filter by the scaled L2 norm of kernel weights.
    Reference: arXiv:1611.06440
    '''
    NO_MASKS = False  # 标志变量，指示是否需要构造新的掩码
    # 如果传入的masks为空，则需要构造新的掩码
    if not masks:
        masks = []
        NO_MASKS = True

    values = []  # 存储每层每个滤波器的重要性得分
    for p in model.parameters():

        if len(p.data.size()) == 4:  # 选择卷积层（假设只有卷积层有四维参数）
            p_np = p.data.cpu().numpy()  # 获取当前层权重的numpy版本

            # 如果需要构造新的掩码，则初始化该层的掩码
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # 计算每个滤波器的缩放L2范数
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1).sum(axis=1) / (
                    p_np.shape[1] * p_np.shape[2] * p_np.shape[3])
            # 对重要性得分进行归一化处理
            value_this_layer = value_this_layer / np.sqrt(np.square(value_this_layer).sum())
            # 找到通道值的最小值以及索引
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])  # 添加到values列表中

    assert len(masks) == len(values), "something wrong here"  # 确保掩码数量与计算的重要性得分数量一致

    values = np.array(values)  # 将values列表转换为NumPy数组

    # 设置对应于要剪枝的滤波器的掩码
    to_prune_layer_ind = np.argmin(values[:, 0])  # 找到具有最小重要性得分的层
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])  # 找到该层中要剪枝的滤波器索引
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.  # 将该滤波器对应的掩码位置设为0

    return masks  # 返回更新后的掩码列表


def filter_prune(model, pruning_perc):
    '''
    Iteratively prune filters until the desired pruning percentage is reached.
    '''
    masks = []  # 初始化掩码列表
    current_pruning_perc = 0.  # 当前的剪枝率

    while current_pruning_perc < pruning_perc:  # 循环直到达到所需的剪枝率
        masks = prune_one_filter(model, masks)  # 剪枝一个滤波器，并更新掩码
        model.set_masks(masks)  # 应用更新后的掩码到模型
        current_pruning_perc = prune_rate(model, verbose=False)  # 计算当前的剪枝率

    return masks  # 返回最终的掩码列表


def plot_weights(model):
    """Plot the distribution of non-zero weights for layers that have weights."""
    modules = [module for module in model.modules()]  # 获取模型中的所有模块
    num_sub_plot = 0  # 子图编号

    for i, layer in enumerate(modules):  # 遍历所有模块
        if hasattr(layer, 'weight'):  # 检查模块是否有权重属性
            plt.subplot(131 + num_sub_plot)  # 创建子图
            w = layer.weight.data  # 获取当前层的权重数据
            w_one_dim = w.cpu().numpy().flatten()  # 将权重展平为一维数组
            plt.hist(w_one_dim[w_one_dim != 0], bins=50)  # 绘制非零权重的直方图
            num_sub_plot += 1  # 更新子图编号

    plt.show()  # 显示所有子图


if __name__ == '__main__':
    a = np.random.randn(3, 5)  # 创建一个3x5的随机数组
    print(abs(a).flatten())  # 打印该数组的绝对值并展平为一维数组
    b = np.percentile(abs(a).flatten(), 20)  # 计算20%分位数
    print(b)  # 打印20%分位数
    b = np.percentile(abs(a).flatten(), 40)  # 计算40%分位数
    print(b)  # 打印40%分位数
    b = np.percentile(abs(a).flatten(), 60)  # 计算60%分位数
    print(b)  # 打印60%分位数
    b = np.percentile(abs(a).flatten(), 80)  # 计算80%分位数
    print(b)  # 打印80%分位数