import torch
import torch.nn.functional as F

def bin2dec(b, bits):
    """
    Convert binary tensor to decimal.
    :param b: Binary tensor
    :param bits: Number of bits
    :return: Decimal tensor
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=b.device, dtype=b.dtype)
    return torch.sum(mask * b, dim=-1)

def sample_gumbel(shape, eps=1e-20, device='cuda'):
    """
    Sample from Gumbel distribution.
    :param shape: Shape of the output tensor
    :param eps: Small value to avoid log(0)
    :param device: Device to place the tensor
    :return: Gumbel samples
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """
    Sample from Gumbel-Softmax distribution.
    :param logits: Logits tensor
    :param temperature: Temperature parameter
    :return: Gumbel-Softmax sample
    """
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    """
    Apply Gumbel-Softmax with optional hard sampling.
    :param logits: Logits tensor
    :param temperature: Temperature parameter
    :param hard: Whether to return a one-hot vector
    :return: Gumbel-Softmax result
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
        return (y_hard - y).detach() + y
    return y

class LUTLayer(torch.nn.Module):
    """
    LUT Layer.
    """
    def __init__(self, input_dim, lut_size=2, n_luts=1, connections='random', device='cpu'):
        """
        :param input_dim: Number of input dimensions
        :param lut_size: Number of inputs per LUT
        :param n_luts: Number of LUTs
        :param connections: Method for initializing connectivity
        :param device: Device to place the tensors
        """
        super().__init__()
        self.n_luts = n_luts
        self.lut_size = lut_size
        self.w = torch.nn.Parameter(torch.randn(n_luts, 2 ** lut_size, device=device))
        self.w_comp = torch.nn.Parameter(torch.randn(n_luts, 2 ** lut_size, device=device))
        self.indices = self.get_connections(input_dim, lut_size, n_luts, connections, device)

    def forward(self, x):
        inputs = [x[..., idx] for idx in self.indices]
        x_reordered = torch.stack(inputs, dim=-1)
        logits = torch.stack((self.w, self.w_comp), dim=0)

        if self.training:
            noise = 0.0 * torch.randn_like(logits)
            sigma_w_probs = F.softmax((logits + noise) / 1, dim=0)
            sigma_w = sigma_w_probs[0]
            probs = self.get_probs(x_reordered)
            output = (probs * sigma_w).sum(-1)
        else:
            idx = bin2dec(torch.stack(inputs, dim=-1).round().int(), self.lut_size)
            rounded_luts = torch.round(F.softmax(logits, dim=0)[0])
            output = rounded_luts[range(rounded_luts.shape[0]), idx]

        return output

    def get_connections(self, input_dim, lut_input, n_luts, connections, device):
        """
        Get LUT connections.
        :param input_dim: Number of input dimensions
        :param lut_input: Number of inputs per LUT
        :param n_luts: Number of LUTs
        :param connections: Method for initializing connectivity
        :param device: Device to place the tensors
        :return: Connection indices
        """
        if connections == 'random':
            conn = torch.randperm(lut_input * n_luts) % input_dim
            conn = torch.randperm(input_dim)[conn]
            conn = conn.reshape(lut_input, n_luts).to(device)
            return conn
        else:
            raise ValueError("Invalid connection method: {}".format(connections))

    @staticmethod
    def get_probs(x):
        """
        Compute probabilities from inputs.
        :param x: Input tensor
        :return: Probability tensor
        """
        x = x[..., None]
        x_ = torch.cat([x, 1 - x], dim=-1)
        b, neur, n_components, _ = x.shape

        temp = torch.einsum('bnp,bnq->bnpq', x_[:, :, 0, :], x_[:, :, 1, :])
        for i in range(2, n_components):
            temp = torch.einsum('bnp,bnq->bnpq', temp.reshape(b, neur, -1), x_[:, :, i, :])

        return temp.reshape(b, neur, -1).flip(-1)

    def extra_repr(self):
        return 'num_luts={}, lut_size={}'.format(self.n_luts, self.lut_size)

class Aggregation(torch.nn.Module):
    """
    Aggregation module to aggregate outputs.
    """
    def __init__(self, num_classes: int, tau: float = 1.):
        """
        :param num_classes: Number of intended real valued outputs
        :param tau: Softmax temperature
        """
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x):
        """
        Aggregate the output tensor.
        :param x: Input tensor
        :return: Aggregated tensor
        """
        return x.reshape(x.size(0), self.num_classes, -1).sum(-1) / self.tau

    def extra_repr(self):
        return 'num_classes={}, tau={}'.format(self.num_classes, self.tau)