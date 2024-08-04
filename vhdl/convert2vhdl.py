import sys
import os
import argparse
import torch
from pathlib import Path
import numpy as np

from torch.nn import Flatten
import torch.nn.functional as F
sys.path.append(str(Path(__file__).parent.parent))
from lutnn.lutlayer import LUTLayer, Aggregation


def get_args():
    parser = argparse.ArgumentParser(description='Convert LUTNN to VHDL code.')
    parser.add_argument('--model', type=str, required=True, help='LUTNN model name')
    parser.add_argument('--name', type=str, help='Output VHDL name')
    return parser.parse_args()


def create_vhdl_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def gen_globals_file(number_of_inputs, number_of_layers, num_neurons, lut_size, outputs_per_class, output_bits, vhdl_path):
    with open(os.path.join(vhdl_path, "Globals.vhd"), "w") as file:
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('-- Globals.vhd: Package with Components and Constants\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('library IEEE;\n')
        file.write('use IEEE.STD_LOGIC_1164.ALL;\n\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('-- PACKAGE\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('package Globals is\n')
        file.write('\t-- Constants\n')
        file.write(f'\tconstant NET_INPUTS : integer := {number_of_inputs};\t\t-- Number of Inputs of the Logic Net\n')
        for i in range(number_of_layers):
            file.write(f'\tconstant L{i}_NEURONS : integer := {num_neurons[i]};\t\t-- Number of Neurons in Layer {i}\n')
        file.write(f'\tconstant CLASS_OUTS : integer := {outputs_per_class};\t\t-- Number of Binary Outputs per Class\n')
        file.write(f'\tconstant NET_OUTPUT : integer := {output_bits};\t\t\t-- Number of Bits to Represent the Output Classes\n')
        file.write('\n\t-- Layer Component Declarations\n')
        for i in range(number_of_layers):
            if i == 0:
                file.write(f'\tcomponent layer{i} is port ( L{i}_IN : in STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0); L{i}_OUT : out STD_LOGIC_VECTOR(L{i}_NEURONS-1 downto 0) ); end component;\n')
            else:
                file.write(f'\tcomponent layer{i} is port ( L{i}_IN : in STD_LOGIC_VECTOR(L{i-1}_NEURONS-1 downto 0); L{i}_OUT : out STD_LOGIC_VECTOR(L{i}_NEURONS-1 downto 0) ); end component;\n')
        file.write('\n\t-- Comparator Component Declaration\n')
        file.write('\tcomponent comparator is port ( in1, in2 : in natural range 0 to CLASS_OUTS; max : out natural range 0 to CLASS_OUTS; idx : out STD_LOGIC ); end component;\n')
        file.write('\n\t-- Function Declaration\n')
        file.write('\tfunction count_ones(s : STD_LOGIC_VECTOR) return natural;\n')
        file.write('end package;\n\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('-- PACKAGE BODY\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('package body Globals is\n')
        file.write('\tfunction count_ones(s : STD_LOGIC_VECTOR) return natural is\n')
        file.write('\t\tvariable temp : natural := 0;\n')
        file.write('\tbegin\n')
        file.write('\t\tfor i in s\'range loop\n')
        file.write('\t\t\tif s(i) = \'1\' then temp := temp + 1;\n')
        file.write('\t\t\tend if;\n')
        file.write('\t\tend loop;\n')
        file.write('\t\treturn temp;\n')
        file.write('\tend function count_ones;\n')
        file.write('end package body;\n')


def gen_top_file(vhdl_path, number_of_layers, number_of_classes, num_neurons, outputs_per_class, output_bits):
    with open(os.path.join(vhdl_path, "top.vhd"), "w") as file:
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('-- Top.vhd: Top Entity of the Logic Network\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('library IEEE;\n')
        file.write('use IEEE.STD_LOGIC_1164.ALL;\n')
        file.write('use work.Globals.ALL;\n\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('-- ENTITY\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('entity top is\n')
        file.write('\tPort ( NET_I : in  STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0);\t\t-- Input\n')
        file.write('\t\t   NET_O : out STD_LOGIC_VECTOR(NET_OUTPUT-1 downto 0)\t\t-- Output\n')
        file.write('\t);\n')
        file.write('end top;\n\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('-- ARCHITECTURE\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('architecture Structural of top is\n')
        file.write('\t-- Internal Signals Declaration\n')
        for i in range(number_of_layers):
            file.write(f'\tsignal F_L{i} : STD_LOGIC_VECTOR(L{i}_NEURONS-1 downto 0);\n')
        for i in range(number_of_classes):
            file.write(f'\tsignal C{i}   : natural range 0 to CLASS_OUTS;\n')
        for i in range(number_of_classes - 2):
            file.write(f'\tsignal max{i} : natural range 0 to CLASS_OUTS;\n')
        for i in range(number_of_classes - 1):
            file.write(f'\tsignal idx{i} : STD_LOGIC;\n')
        file.write('begin\n')
        file.write('\t-- Instantiate the Layers\n')
        for i in range(number_of_layers):
            if i == 0:
                file.write(f'\tL{i} : layer{i} port map( ')
                file.write(f'L{i}_IN => NET_I, ')
                file.write(f'L{i}_OUT => F_L{i} );\n')
            else:
                file.write(f'\tL{i} : layer{i} port map( ')
                file.write(f'L{i}_IN => F_L{i-1}, ')
                file.write(f'L{i}_OUT => F_L{i} );\n')
        file.write('\n\t-- Sum of Each Output Class (Counting Ones)\n')
        for i in range(number_of_classes):
            file.write(f'\tC{number_of_classes-i-1} <= count_ones(F_L{number_of_layers-1}({num_neurons[-1]-1-outputs_per_class*i} downto {num_neurons[-1]-outputs_per_class-outputs_per_class*i}));\n')
        file.write('\n\t-- Binary Tree Comparisons\n')
        for i in range(number_of_classes-1):
            if i == 0:
                if number_of_classes != 2:
                    file.write('\tCMP0 : comparator port map( in1 => C0, in2 => C1, max => max0, idx => idx0 );\n')
                else:
                    file.write('\tCMP0 : comparator port map( in1 => C0, in2 => C1, max => open, idx => idx0 );\n')
            elif 0 < i < (number_of_classes - 2):
                file.write(f'\tCMP{i} : comparator port map( in1 => max{i-1}, in2 => C{i+1}, max => max{i}, idx => idx{i} );\n')
            else:
                file.write(f'\tCMP{i} : comparator port map( in1 => max{i-1}, in2 => C{i+1}, max => open, idx => idx{i} );\n')
        file.write('\n\t-- Connect Net Output\n')
        for i in range(number_of_classes):
            binchars = f'{i:b}'.zfill(output_bits)
            if i == 0:
                file.write(f'\tNET_O <= "{binchars}" when (')
                print_idx(file, i, number_of_classes)
                file.write(') else\n')
            elif i < (number_of_classes-1):
                file.write(f'\t\t\t "{binchars}" when (')
                print_idx(file, i, number_of_classes)
                file.write(') else\n')
            else:
                file.write(f'\t\t\t "{binchars}" when (')
                print_idx(file, i, number_of_classes)
                file.write(') else\n')
                binchars = f'{2**output_bits-1:b}'.zfill(output_bits)
                file.write(f'\t\t\t "{binchars}";\n')
        file.write('end Structural;\n')


def print_idx(file, comparator, n_idx):
    for i in range(n_idx-1):
        if comparator == 0:
            if i == (n_idx-2):
                file.write(f'idx{i} = \'0\'')
            else:
                file.write(f'idx{i} = \'0\' and ')
        else:
            if i == (comparator-1):
                if i < (n_idx - 2):
                    file.write(f'idx{i} = \'1\' and ')
                else:
                    file.write(f'idx{i} = \'1\'')
            elif (comparator - 1) < i < (n_idx - 2):
                file.write(f'idx{i} = \'0\' and ')
            elif i == (n_idx - 2):
                file.write(f'idx{i} = \'0\'')


def gen_comparator_file(vhdl_path):
    with open(os.path.join(vhdl_path, "comparator.vhd"), "w") as file:
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('-- Comparator.vhd: Receives Two Inputs and Returns the Greatest\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('library IEEE;\n')
        file.write('use IEEE.STD_LOGIC_1164.ALL;\n')
        file.write('use work.Globals.ALL;\n\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('-- ENTITY\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('entity comparator is\n')
        file.write('\tPort ( in1 : in  natural range 0 to CLASS_OUTS;\t\t-- Input 1\n')
        file.write('\t\t   in2 : in  natural range 0 to CLASS_OUTS;\t\t-- Input 2\n')
        file.write('\t\t   max : out natural range 0 to CLASS_OUTS;\t\t-- Greatest Input\n')
        file.write('\t\t   idx : out STD_LOGIC\t\t\t\t\t\t\t-- Greatest Input Index\n')
        file.write('\t);\n')
        file.write('end comparator;\n\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('-- ARCHITECTURE\n')
        file.write('-----------------------------------------------------------------------------------------\n')
        file.write('architecture Behavioral of comparator is\n')
        file.write('begin\n')
        file.write('\tprocess (in1, in2) begin\n')
        file.write('\t\tif (in1 >= in2) then\n')
        file.write('\t\t\tidx <= \'0\';\n')
        file.write('\t\t\tmax <= in1;\n')
        file.write('\t\telse\n')
        file.write('\t\t\tidx <= \'1\';\n')
        file.write('\t\t\tmax <= in2;\n')
        file.write('\t\tend if;\n')
        file.write('\tend process;\n')
        file.write('end Behavioral;\n')


def gen_layer_header(file, layer, lut_size):
    file.write('-----------------------------------------------------------------------------------------\n')
    file.write(f'-- Layer{layer}.vhd: Layer {layer}\n')
    file.write('-----------------------------------------------------------------------------------------\n')
    file.write('library IEEE;\n')
    file.write('use IEEE.STD_LOGIC_1164.ALL;\n')
    file.write('use work.Globals.ALL;\n\n')
    file.write('library UNISIM;\n')
    file.write('use UNISIM.vcomponents.ALL;\n\n')
    file.write('-----------------------------------------------------------------------------------------\n')
    file.write('-- ENTITY\n')
    file.write('-----------------------------------------------------------------------------------------\n')
    file.write(f'entity layer{layer} is\n')
    file.write('\tPort ( ')
    if layer == 0:
        file.write(f'L{layer}_IN  : in  STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0);\t-- Layer input\n\t\t   ')
    else:
        file.write(f'L{layer}_IN  : in  STD_LOGIC_VECTOR(L{layer - 1}_NEURONS-1 downto 0);\t-- Layer input\n\t\t   ')
    file.write(f'L{layer}_OUT : out STD_LOGIC_VECTOR(L{layer}_NEURONS-1 downto 0)\t\t-- Layer output\n')
    file.write('\t);\n')
    file.write(f'end layer{layer};\n\n')
    file.write('-----------------------------------------------------------------------------------------\n')
    file.write('-- ARCHITECTURE\n')
    file.write('-----------------------------------------------------------------------------------------\n')
    file.write(f'architecture Structural of layer{layer} is\n')
    file.write('begin\n')
    file.write('\t-- Instantiate the Neurons\n')
    return file


def process_file(layers, vhdl_path, num_neurons, lut_size):
    for l, layer in enumerate(layers):
        indices, values = layer
        input_neurons = indices.max().item()
        with open(os.path.join(vhdl_path, f"layer{l}.vhd"), "w") as file:
            file = gen_layer_header(file, l, lut_size)
            for neuron in range(num_neurons[l]):
                file.write(f'\tN{neuron}_L{l} : LUT{lut_size[l]} generic map (INIT => X"{hex(values[neuron])[2:].zfill((2**lut_size[l]) // 4)}") port map(')
                for i in range(lut_size[l]):
                    file.write(f'I{lut_size[l]-i-1} => L{l}_IN({indices[i][neuron]}), ')
                file.write(f'O => L{l}_OUT({neuron}));\n')
            file.write('end Structural;\n')


def gen_vhdl_code(input_data, name, number_of_layers, number_of_classes, number_of_inputs, num_neurons, lut_size):
    data_path = "data"
    vhdl_path = os.path.join(data_path, "VHDL", name)

    output_bits = len(f'{number_of_classes-1:b}')
    outputs_per_class = num_neurons[-1] // number_of_classes

    create_vhdl_folder(vhdl_path)
    gen_globals_file(number_of_inputs, number_of_layers, num_neurons, lut_size, outputs_per_class, output_bits, vhdl_path)
    gen_top_file(vhdl_path, number_of_layers, number_of_classes, num_neurons, outputs_per_class, output_bits)
    gen_comparator_file(vhdl_path)

    layers = get_net_layers(input_data, lut_size)
    process_file(layers, vhdl_path, num_neurons, lut_size)


def get_prefix_sums(layers):
    prefix_sums = [0]
    cur_count = 0
    for layer_a, layer_b, layer_op in layers[:-1]:
        cur_count += len(layer_a)
        prefix_sums.append(cur_count)
    return prefix_sums


def get_net_layers(model, verbose=False):
    layers = []
    first = True
    print(model.model)
    idx = -1
    for layer in model.model:
        if isinstance(layer, LUTLayer):
            idx += 1
            if first:
                first = False
            logits = torch.stack((layer.w, layer.w_comp), dim=0)
            w_round = torch.round(F.softmax(logits, dim=0)[0]).type(torch.int64)
            print(w_round)
            value = 0
            for i in range(1, (2**lut_size[idx])+1):
                value += np.array(w_round[:, i-1].cpu(), dtype=np.uint) * 2**(i-1)
            layers.append((layer.indices, value))
            # layers.append((layer.indices[0], layer.indices[1], value))#w_round[:,0] * 8 + w_round[:,1] * 4 + w_round[:,2] * 2 + w_round[:,3]))
        elif isinstance(layer, Flatten):
            if verbose:
                print('Skipping torch.nn.Flatten layer ({}).'.format(type(layer)))
        elif isinstance(layer, Aggregation):
            if verbose:
                print('Skipping GroupSum layer ({}).'.format(type(layer)))
        else:
            assert False, 'Error: layer {} / {} unknown.'.format(type(layer), layer)
    return layers


def get_model_params(model):
    lut_size = []
    num_neurons = []
    number_of_inputs = -1
    for layer in model.model:
        if isinstance(layer, LUTLayer):
            if number_of_inputs == -1:
                number_of_inputs = torch.max(layer.indices).item() + 1
            lut_size.append(layer.indices.shape[0])
            num_neurons.append(layer.indices.shape[1])
        elif isinstance(layer, Aggregation):
            number_of_classes = layer.num_classes
    number_of_layers = len(num_neurons)
    return number_of_layers, num_neurons, lut_size, number_of_inputs, number_of_classes


if __name__ == "__main__":
    args = get_args()
    if args.name is None:
        args.name = args.model

    model = torch.load(f"models/{args.model}.pth")
    number_of_layers, num_neurons, lut_size, number_of_inputs, number_of_classes = get_model_params(model)
    gen_vhdl_code(model, args.name, number_of_layers, number_of_classes, number_of_inputs, num_neurons, lut_size)
