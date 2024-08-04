<img src="https://github.com/user-attachments/assets/3c150f9b-339d-4299-bb97-12ea93d8f0cb" alt="image" style="width:18%;">

# LUTNN - A Look-Up Table Neural Network trainable architecture

![g_abstract](https://github.com/user-attachments/assets/5fc1eb5d-7601-40f2-a293-b35771294d76)


This repository implements Look-Up Tables Neural Networks (LUTNNs)


## Installation
`pip install -e .`

## Training example

```python
import torch
from lutnn.lutlayer import LUTLayer, Aggregation

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    LUTLayer(input_dim=1000, lut_size=6, n_luts=2048),
    LUTLayer(input_dim=2048, lut_size=6, n_luts=4000),
    Aggregation(num_classes=10, tau = 10)
)
```
## VHDL


