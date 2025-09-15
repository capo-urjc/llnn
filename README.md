<img src="https://github.com/user-attachments/assets/7c621714-ee04-443a-bb0b-b76cabb5b4a5" alt="image" style="width:18%;">

# LLNN: A Scalable LUT-Based Logic Neural Network Architecture for FPGAs

<img width="3180" height="1974" alt="model_comparison_avg-1" src="https://github.com/user-attachments/assets/59340012-51a4-49fc-b311-68f6f33bbc87" />



This repository implements Look-Up Tables Logic Neural Networks (LLNNs) presented in:

[LLNN: A Scalable LUT-Based Logic Neural Network Architecture for FPGAs](https://ieeexplore.ieee.org/abstract/document/11154450)

![g_abstract](https://github.com/user-attachments/assets/1f85e6a0-8a10-417d-bc6e-a6e793369c41)

## 📖 Citing

If you use this work, please cite the following paper:

```bibtex
@ARTICLE{11154450,
  author={Ramírez, Iván and Garcia-Espinosa, Francisco J. and Concha, David and Aranda, Luis Alberto and Schiavi, Emanuele},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers}, 
  title={LLNN: A Scalable LUT-Based Logic Neural Network Architecture for FPGAs}, 
  year={2025},
  volume={},
  number={},
  pages={1-13},
  keywords={Logic gates;Field programmable gate arrays;Biological neural networks;Table lookup;Hardware;Computer architecture;Training;Neurons;Logic;Complexity theory;Logic neural networks;LUT-based neural networks;FPGA implementation;scalability in hardware neural networks;real-time inference},
  doi={10.1109/TCSI.2025.3606054}
}
```

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

We offer a testbench with VHDL code and additional utility functions (e.g., `bin_to_hex`) for validating a LLNN trained on the MNIST dataset.
