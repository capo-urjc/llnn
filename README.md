<img src="https://github.com/user-attachments/assets/3c150f9b-339d-4299-bb97-12ea93d8f0cb" alt="image" style="width:18%;">

# LUTNN - A Look-Up Table Neural Network trainable architecture

![g_abstract](https://github.com/user-attachments/assets/5fc1eb5d-7601-40f2-a293-b35771294d76)


This repository implements Look-Up Tables Neural Networks (LUTNNs)

## Training example

```python
from lutnn import lutlayer, aggregation
import torch

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    lutlayer(784, 16_000),
    lutlayer(16_000, 16_000),
    aggregation(k=10, tau=30)
)
```
## VHDL


