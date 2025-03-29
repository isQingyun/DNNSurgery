import torch.nn  as nn

def is_primary_layer(layer):
    """判断是否为基础计算层"""
    return isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                              nn.Linear,
                              nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                              nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d))


def is_auxiliary_layer(layer):
    """判断是否为辅助层"""
    return isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh,
                              nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d))

def is_adaptive(layer):
    """判断是否为自适应层"""
    return isinstance(layer, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,))


def merge_layers(model):
    """合并基础层与相邻辅助层"""
    # 提取所有叶子层（无子模块的层）
    leaf_layers = []
    for module in model.modules():
        if not list(module.children()):
            leaf_layers.append(module)

    merged = []
    i = 0
    while i < len(leaf_layers):
        layer = leaf_layers[i]
        if is_primary_layer(layer):
            # 创建新容器，包含基础层及其后的辅助层
            container = [layer]
            j = i + 1
            while j < len(leaf_layers) and is_auxiliary_layer(leaf_layers[j]):
                container.append(leaf_layers[j])
                j += 1
            merged.append(nn.Sequential(*container))
            i = j  # 跳过已处理的辅助层
        elif is_adaptive(layer):
            container = [layer, nn.Flatten()]
            merged.append(nn.Sequential(*container))
            i += 1
        else:
            merged.append(layer)
            i += 1
    return merged