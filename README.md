<img src="https://github.com/DavidConcha/DiffLUTs/assets/17121671/024f0078-bb97-4476-a44c-005191a7f7c0" alt="image" style="width:18%;">

# LUTNN - A Look-Up Table Neural Network trainable architecture

![image](https://github.com/DavidConcha/DiffLUTs/assets/17121671/302019cc-f565-4381-bc74-a83027997319)


This repository implements Look-Up Tables Neural Networks (LUTNNs)

## Training example

```python
from lnn import LogicLayer, GroupSum
import torch

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    LogicLayer(784, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    GroupSum(k=10, tau=30)
)
```
## VHDL


