import torch
import torchvision
from Models.exitModel import ExitModel
from Models.entryModel import EntryModel
from Utilities import utilities

# 原始网络 此处以VGG16为例
vgg16 = torchvision.models.vgg16(pretrained = True)

# 实例化edge model, 可动态中断
exitmodel = ExitModel(vgg16)
# 实例化cloud model, 可动态进入
entrymodel = EntryModel(vgg16)

# 确认断点位置，不大于split后的模型层数
breakpoint = 9
input = torch.randn(1,3,224,224)

# 查看网络结构
# print(vgg16)
# print(len(exitmodel.layers))
# print(exitmodel.layers)

# 注册钩子获取每层输出张量的数据大小，可用于动态分割点的确认
# exitmodel_output = Utilities.get_layer_output_shapes(entrymodel,torch.randn(1,3,224,224))
# print(exitmodel_output)

# edge model运行
feature = exitmodel(input, exit_layer = breakpoint)
# cloud model运行
result = entrymodel(feature, entry_layer = breakpoint+1)

# 导出并保存当前的feature/result
utilities.save_tensor(feature, "feature")
utilities.save_tensor(result, "result")