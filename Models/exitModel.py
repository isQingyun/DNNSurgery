from torch import nn
from Utilities.splitModel import merge_layers

class ExitModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        base_layers = merge_layers(original_model)
        self.layers = nn.ModuleList(base_layers)

    def forward(self, x, exit_layer):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if exit_layer == idx:
                print(f"Exited at layer {exit_layer}, Intermediate shape: {x.shape}")
                break
        return x
