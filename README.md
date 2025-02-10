<img src="https://github.com/user-attachments/assets/858cda97-d734-4faf-946a-88f9972b9334" alt="image" style="width:18%;">

# LLNN - A Look-Up Table Logic Neural Network trainable architecture

![g_abstract](https://github.com/user-attachments/assets/1f85e6a0-8a10-417d-bc6e-a6e793369c41)

This repository implements Look-Up Tables Logic Neural Networks (LLNNs)


## Installation
`pip install -e .`

## Importing LUTLayer

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

## Train a model

`python main.py --train --save --name model1 --dataset mnist --batch-size 128 -lr 0.01 --num-iterations 10000`

## Test a trained model

`python main.py --load --name model1 --dataset mnist`


# VHDL

![Toolflow 1](https://github.com/user-attachments/assets/2e751f7c-c13d-48fd-9776-e09ff8ce25f3)

## VHDL code generation example

` python vhdl/convert2vhdl.py --model model1`

## LUTNN Testbench

We offer a testbench with VHDL code and additional utility functions (e.g., `bin_to_hex`) for validating a LUTNN trained on the MNIST dataset.
