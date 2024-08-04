import torch
import torch.nn.functional as F
import numpy as np

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=0)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    return y


class LUTLayer(torch.nn.Module):
    """
    LUT layer.
    """
    def __init__(self, input_dim, lut_size=2, n_luts=1, connections: str = 'random', device='cpu'):

        """
        :param lut_size:      Number of inputs per LUT
        :param n_luts:     Number of LUTs
        :param connections: method for initializing the connectivity of the logic gate net
        """

        super().__init__()

        self.n_luts = n_luts
        self.w_comp = torch.nn.parameter.Parameter(torch.randn(n_luts, 2 ** lut_size, device=device))
        self.w = torch.nn.parameter.Parameter(torch.randn(n_luts, 2 ** lut_size, device=device))
        self.n_inputs = lut_size
        self.indices = self.get_connections(input_dim, lut_size, n_luts, connections, device)

    def forward(self, x):
        inputs = [x[..., idx] for idx in self.indices]
        x_reordered = torch.stack(inputs, dim=-1)
        logits = torch.stack((self.w, self.w_comp), dim=0)

        if self.training:
            # Apply the LUT activation function
            noise = 0.0 * torch.randn_like(logits)
            sigma_w_probs = F.softmax((logits + noise) /1, dim=0)

            sigma_w = (1 * sigma_w_probs[0]) + (0 * sigma_w_probs[1])

            probs = self.get_probs(x_reordered)
            output = (probs * sigma_w).sum(-1)

        else:
            idx = bin2dec(torch.stack(inputs, dim=-1).round().int(), self.n_inputs)
            rounded_luts = torch.round(F.softmax(logits, dim=0)[0])

            indices = list(range(rounded_luts.shape[0]))
            output = rounded_luts[indices, idx]

        return output

    def get_connections(self, input_dim, lut_input, n_luts, connections, device):
        # if connections == 'random':
        #     lut_connections = lut_input * n_luts
        #     n_tiles = int(np.ceil((lut_connections / input_dim)))
        #     c = torch.randperm(input_dim).tile(n_tiles)[0:lut_connections]
        #     c = c.reshape(lut_input, n_luts).to(device)
        #     return c
        if connections == 'random':
            conn = torch.randperm(lut_input * n_luts) % input_dim
            conn = torch.randperm(input_dim)[conn]
            conn = conn.reshape(lut_input, n_luts).to(device)
            return conn
        else:
            raise ValueError(connections)

    @staticmethod
    def get_probs(x):
        x = x[..., None]
        x_ = torch.cat([x, 1 - x], dim=-1)

        b, neur, n_components, _ = x.shape

        for i in range(n_components - 1):

            if i == 0:
                temp = torch.einsum('bnp,bnq->bnpq', x_[:, :, i, :], x_[:, :, i + 1, :])
            else:
                temp = torch.einsum('bnp,bnq->bnpq', temp.reshape(b, neur, -1), x_[:, :, i + 1, :])

        result = temp.reshape(b, neur, -1).flip(-1)
        return result

    def extra_repr(self):
        return 'num_luts={}, lut_size={}'.format(self.n_luts, self.n_inputs)


class Aggregation(torch.nn.Module):
    """
    The Aggregation module.
    """
    def __init__(self, num_classes: int, tau: float = 1.):
        """

        :param num_classes: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x):
        return x.reshape(x.size(0), self.num_classes, int(x.size(-1) / self.num_classes)).sum(-1) / self.tau

    def extra_repr(self):
        return 'num_classes={}, tau={}'.format(self.num_classes, self.tau)
