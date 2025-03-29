import torch
from torch import nn
import pandas as pd
import numpy as np
from collections import OrderedDict

def get_mem_size(tensor):
    return tensor.element_size() * tensor.numel()

def get_layer_output_shapes(model, input_tensor):
    outputs = OrderedDict()

    # 定义前向钩子函数
    def output_hook(module, input, output):
        name = f"{type(module).__name__}[{len(outputs) + 1}]"
        outputs[name] = (output.shape,get_mem_size(output))

    # 递归注册钩子
    def register_hooks(module):
        hooks = []
        if len(list(module.children())) == 0:  # 叶子节点
            hook = module.register_forward_hook(output_hook)
            hooks.append(hook)
        else:
            for child in module.children():
                register_hooks(child)
        return hooks

    hooks = register_hooks(model)

    # 运行前向传播
    with torch.no_grad():
        model(input_tensor)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    return outputs

def save_tensor(tensor,name):
    print(tensor)
    tensor_np = tensor.detach().cpu().numpy()
    np.save(f"./TensorOutput/{name}_tensor.npy", tensor_np)
    tensor_np = tensor_np.flatten()
    pd.DataFrame(tensor_np).to_csv(f"./TensorOutput/{name}_tensor.csv", header=False, index=False)