import torch
from lutnn.lutlayer import LUTLayer, Aggregation
from torch import nn

class LUTNN(torch.nn.Module):
    def __init__(self, luts_per_layer, num_layers, lut_size, input_dim: int, num_classes: int, device="cpu",
                 connections='random', tau=10):
        super(LUTNN, self).__init__()

        if len(lut_size) < num_layers:
            lut_size = [lut_size[0]] * num_layers
        if len(luts_per_layer) < num_layers:
            luts_per_layer = [luts_per_layer[0]] * num_layers
        if luts_per_layer[-1] % 10 != 0:
            print("The number of luts in the last layer must be multiple of num_classes")
            prev_luts = luts_per_layer[-1]
            luts_per_layer[-1] = round(luts_per_layer[-1] / num_classes) * num_classes
            print(f"Using {luts_per_layer[-1]} instead of {prev_luts} luts")
        layers = []
        if connections == 'random':
            layers.append(torch.nn.Flatten())
            layers.append(LUTLayer(input_dim=input_dim, lut_size=lut_size[0], n_luts=luts_per_layer[0], connections=connections, device=device))
            for i in range(1, num_layers):
                layers.append(LUTLayer(input_dim=luts_per_layer[i-1], lut_size=lut_size[i], n_luts=luts_per_layer[i], connections=connections, device=device))
            layers.append(Aggregation(num_classes, tau))
            self.model = torch.nn.Sequential(
                *layers
            )
        else:
            raise NotImplementedError(connections)

        print(f'Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(self.model)

    def forward(self, x):
        y = self.model(x)
        return y
