from torch import nn
from Utilities.splitModel import merge_layers

class EntryModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        base_layers = merge_layers(original_model)
        self.layers = nn.ModuleList(base_layers)

    def forward(self, x, entry_layer):
        assert len(self.layers) > entry_layer >= 0
        print(f"Entery at layer {entry_layer}, Intermediate shape: {x.shape}")
        for layer in self.layers[entry_layer:]:
            x = layer(x)
        return x