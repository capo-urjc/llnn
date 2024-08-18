from torch import nn
from lutnn.lutlayer import LUTLayer, Aggregation


class LUTNN(nn.Module):
    def __init__(self, luts_per_layer, num_layers, lut_size, input_dim: int, num_classes: int, device='cpu',
                 connections='random', tau=10):
        """
        Initialize the LUTNN model.

        :param luts_per_layer: List of number of LUTs per layer
        :param num_layers: Number of layers
        :param lut_size: List of LUT sizes per layer
        :param input_dim: Dimension of the input
        :param num_classes: Number of output classes
        :param device: Device to use ('cpu' or 'cuda')
        :param connections: Method for initializing connectivity ('random')
        :param tau: Softmax temperature
        """
        super(LUTNN, self).__init__()

        # Ensure lut_size and luts_per_layer have the correct length
        lut_size = [lut_size] * num_layers if isinstance(lut_size, int) else lut_size
        luts_per_layer = [luts_per_layer] * num_layers if isinstance(luts_per_layer, int) else luts_per_layer

        if len(lut_size) < num_layers:
            lut_size = lut_size + [lut_size[-1]] * (num_layers - len(lut_size))
        if len(luts_per_layer) < num_layers:
            luts_per_layer = luts_per_layer + [luts_per_layer[-1]] * (num_layers - len(luts_per_layer))

        # Ensure the number of LUTs in the last layer is a multiple of num_classes
        if luts_per_layer[-1] % 10 != 0:
            print("Warning: The number of LUTs in the last layer should be a multiple of num_classes.")
            prev_luts = luts_per_layer[-1]
            luts_per_layer[-1] = round(luts_per_layer[-1] / num_classes) * num_classes
            print(f"Adjusted number of LUTs in the last layer to {luts_per_layer[-1]} from {prev_luts}.")

        layers = []
        if connections == 'random':
            layers.append(nn.Flatten())
            layers.append(
                LUTLayer(input_dim=input_dim, lut_size=lut_size[0], n_luts=luts_per_layer[0], connections=connections,
                         device=device))
            for i in range(1, num_layers):
                layers.append(LUTLayer(input_dim=luts_per_layer[i - 1], lut_size=lut_size[i], n_luts=luts_per_layer[i],
                                       connections=connections, device=device))
            layers.append(Aggregation(num_classes=num_classes, tau=tau))
            self.model = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f"Connection method '{connections}' is not implemented.")

        # Print model parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of parameters: {num_params}')
        print(self.model)

    def forward(self, x):
        """
        Forward pass through the model.

        :param x: Input tensor
        :return: Output tensor
        """
        return self.model(x)