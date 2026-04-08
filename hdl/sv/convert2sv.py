import sys
import os
import argparse
import torch
from pathlib import Path
import numpy as np

from torch.nn import Flatten
import torch.nn.functional as F
sys.path.append(str(Path(__file__).parent.parent))
from llnn.lutlayer import LUTLayer, Aggregation


def get_args():
    parser = argparse.ArgumentParser(description="Convert LUTNN to SystemVerilog code.")
    parser.add_argument("--model", type=str, required=True, help="LUTNN model name (path stem in models/)")
    parser.add_argument("--name", type=str, help="Output SV folder name (defaults to --model)")
    return parser.parse_args()

def create_sv_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def gen_globals_file(number_of_inputs, number_of_layers, num_neurons, lut_size, outputs_per_class, output_bits, sv_path):
    with open(os.path.join(sv_path, "Globals.sv"), "w") as file:
        file.write("// -------------------------------------------------------------------------------------\n")
        file.write("// Globals.sv: Parameters for LUTNN\n")
        file.write("// -------------------------------------------------------------------------------------\n")
        file.write("\n")

        file.write("`ifndef GLOBALS_SV\n")
        file.write("`define GLOBALS_SV\n")
        file.write("\n")

        file.write("// Network configuration\n")
        file.write(f"localparam NET_INPUTS      = {number_of_inputs};\n")
        for i in range(number_of_layers):
            file.write(f"localparam L{i}_NEURONS      = {num_neurons[i]};\n")
        file.write(f"localparam CLASS_OUTS      = {outputs_per_class};\n")
        file.write(f"localparam NET_OUTPUT_BITS = {output_bits};\n")

        file.write("\n")
        file.write("`endif // GLOBALS_SV\n")

def gen_top_file(sv_path, number_of_layers, number_of_classes, num_neurons, outputs_per_class, output_bits):
    with open(os.path.join(sv_path, "top.sv"), "w") as file:
        file.write("// --------------------------------------------------------------------------------------\n")
        file.write("// top.sv: Top module of the LUTNN\n")
        file.write("// --------------------------------------------------------------------------------------\n")
        file.write("\n")

        file.write("`include \"Globals.sv\"\n")
        file.write("\n")

        file.write("module top (\n")
        file.write("\tinput  logic [NET_INPUTS-1:0] NET_I,\n")
        file.write(f"\toutput logic [{output_bits-1}:0]            NET_O\n")
        file.write(");\n")
        file.write("\n")

        for i in range(number_of_layers):
            file.write(f"\tlogic [L{i}_NEURONS-1:0]           F_L{i};\n")

        c_width = "$clog2(CLASS_OUTS+1)"
        for i in range(number_of_classes):
            file.write(f"\tlogic [{c_width}-1:0] C{i};\n")

        for i in range(number_of_classes - 2):
            file.write(f"\tlogic [{c_width}-1:0] max{i};\n")

        for i in range(number_of_classes - 1):
            file.write(f"\tlogic                            idx{i};\n")

        file.write("\n")

        file.write("\t// Instantiate layers\n")
        for i in range(number_of_layers):
            file.write(f"\tlayer{i} L{i} (\n")
            if i == 0:
                file.write("\t\t.in  (NET_I),\n")
            else:
                file.write(f"\t\t.in  (F_L{i-1}),\n")
            file.write(f"\t\t.out (F_L{i})\n")
            file.write("\t);\n")
            file.write("\n")

        # Emit a parameterized popcount function
        result_width = (outputs_per_class - 1).bit_length() + 1
        file.write(f"\t// Parameterized popcount function\n")
        file.write(f"\tfunction automatic [{result_width-1}:0] popcount;\n")
        file.write(f"\t\tinput [{outputs_per_class-1}:0] v;\n")
        file.write(f"\t\tinteger i;\n")
        file.write(f"\t\tbegin\n")
        file.write(f"\t\t\tpopcount = 0;\n")
        file.write(f"\t\t\tfor (i = 0; i < {outputs_per_class}; i = i + 1)\n")
        file.write(f"\t\t\t\tpopcount = popcount + v[i];\n")
        file.write(f"\t\tend\n")
        file.write(f"\tendfunction\n")
        file.write("\n")

        file.write("\t// Popcount per class\n")
        for i in range(number_of_classes):
            hi = num_neurons[-1] - 1 - outputs_per_class * i
            lo = num_neurons[-1] - outputs_per_class * (i + 1)
            file.write(f"\tassign C{number_of_classes - i - 1} = popcount(F_L{number_of_layers - 1}[{hi}:{lo}]);\n")


        file.write("\n")

        file.write("\t// Comparator reduction chain\n")
        for i in range(number_of_classes - 1):
            file.write(f"\tcomparator CMP{i} (\n")
            if i == 0:
                file.write("\t\t.in1 (C0),\n")
                file.write("\t\t.in2 (C1),\n")
            else:
                file.write(f"\t\t.in1 (max{i-1}),\n")
                file.write(f"\t\t.in2 (C{i+1}),\n")

            if i < number_of_classes - 2:
                file.write(f"\t\t.max (max{i}),\n")
            else:
                file.write("\t\t.max (),\n")

            file.write(f"\t\t.idx (idx{i})\n")
            file.write("\t);\n")
            file.write("\n")

        file.write("\t// Output encoding (argmax)\n")
        file.write("\talways_comb begin\n")
        file.write(f"\t\tNET_O = {output_bits}'b{'1' * output_bits};\n")

        for i in range(number_of_classes):
            binchars = f"{i:b}".zfill(output_bits)
            file.write("\t\tif (")
            print_idx(file, i, number_of_classes)
            file.write(f") NET_O = {output_bits}'b{binchars};\n")

        file.write("\tend\n")
        file.write("\n")

        file.write("endmodule : top\n")

def print_idx(file, comparator, n_idx):
    for i in range(n_idx - 1):
        if comparator == 0:
            if i == (n_idx - 2):
                file.write(f"idx{i} == 1'b0")
            else:
                file.write(f"idx{i} == 1'b0 && ")
        else:
            if i == (comparator - 1):
                if i < (n_idx - 2):
                    file.write(f"idx{i} == 1'b1 && ")
                else:
                    file.write(f"idx{i} == 1'b1")
            elif (comparator - 1) < i < (n_idx - 2):
                file.write(f"idx{i} == 1'b0 && ")
            elif i == (n_idx - 2):
                file.write(f"idx{i} == 1'b0")


def gen_comparator_file(sv_path):
    with open(os.path.join(sv_path, "comparator.sv"), "w") as file:
        file.write("// --------------------------------------------------------------------------------------\n")
        file.write("// comparator.sv: Receives two inputs and outputs the maximum and its index\n")
        file.write("// --------------------------------------------------------------------------------------\n")
        file.write("\n")

        file.write("`include \"Globals.sv\"\n")
        file.write("\n")

        file.write("module comparator (\n")
        file.write("\tinput  logic [$clog2(CLASS_OUTS+1)-1:0] in1,\n")
        file.write("\tinput  logic [$clog2(CLASS_OUTS+1)-1:0] in2,\n")
        file.write("\toutput logic [$clog2(CLASS_OUTS+1)-1:0] max,\n")
        file.write("\toutput logic                            idx\n")
        file.write(");\n")
        file.write("\n")

        file.write("\talways_comb begin\n")
        file.write("\t\tif (in1 >= in2) begin\n")
        file.write("\t\t\tidx = 1'b0;\n")
        file.write("\t\t\tmax = in1;\n")
        file.write("\t\tend else begin\n")
        file.write("\t\t\tidx = 1'b1;\n")
        file.write("\t\t\tmax = in2;\n")
        file.write("\t\tend\n")
        file.write("\tend\n")
        file.write("\n")

        file.write("endmodule : comparator\n")


def gen_layer_header(file, layer):
    file.write("// --------------------------------------------------------------------------------------\n")
    file.write(f"// layer{layer}.sv: Layer {layer}\n")
    file.write("// --------------------------------------------------------------------------------------\n")
    file.write("\n")

    file.write("`include \"Globals.sv\"\n")
    file.write("\n")

    file.write(f"module layer{layer} (\n")
    if layer == 0:
        file.write("\tinput  logic [NET_INPUTS-1:0] in,\n")
    else:
        file.write(f"\tinput  logic [L{layer - 1}_NEURONS-1:0] in,\n")
    file.write(f"\toutput logic [L{layer}_NEURONS-1:0] out\n")
    file.write(");\n")
    file.write("\n")

    file.write("\t// Instantiate the neurons (LUTs)\n")
    file.write("\n")

    return file


def process_file(layers, sv_path, num_neurons, lut_size):
    for l, layer in enumerate(layers):
        indices, values = layer

        with open(os.path.join(sv_path, f"layer{l}.sv"), "w") as file:
            file = gen_layer_header(file, l)

            for neuron in range(num_neurons[l]):
                file.write(f"\t// Neuron {neuron}\n")
                file.write(f"\tlogic [{lut_size[l]-1}:0] lut_in_{neuron};\n")
                file.write(f"\tlogic lut_out_{neuron};\n")

                for i in range(lut_size[l]):
                    src = indices[i][neuron]
                    file.write(f"\tassign lut_in_{neuron}[{lut_size[l]-i-1}] = in[{src}];\n")

                file.write("\talways_comb begin\n")
                file.write(f"\t\tcase (lut_in_{neuron})\n")

                table = int(values[neuron])
                for k in range(2 ** lut_size[l]):
                    bit = (table >> k) & 1
                    file.write(f"\t\t\t{lut_size[l]}'d{k}: lut_out_{neuron} = 1'b{bit};\n")

                file.write(f"\t\t\tdefault: lut_out_{neuron} = 1'b0;\n")
                file.write("\t\tendcase\n")
                file.write("\tend\n")

                file.write(f"\tassign out[{neuron}] = lut_out_{neuron};\n")
                file.write("\n")

            file.write(f"endmodule : layer{l}\n")

def gen_sv_code(input_data, name, number_of_layers, number_of_classes, number_of_inputs, num_neurons, lut_size):
    data_path = "data"
    sv_path = os.path.join(data_path, "sv", name)

    output_bits = len(f'{number_of_classes-1:b}')
    outputs_per_class = num_neurons[-1] // number_of_classes

    create_sv_folder(sv_path)
    gen_globals_file(number_of_inputs, number_of_layers, num_neurons, lut_size, outputs_per_class, output_bits, sv_path)
    gen_top_file(sv_path, number_of_layers, number_of_classes, num_neurons, outputs_per_class, output_bits)
    gen_comparator_file(sv_path)

    layers = get_net_layers(input_data, lut_size)
    process_file(layers, sv_path, num_neurons, lut_size)

def get_prefix_sums(layers):
    prefix_sums = [0]
    cur_count = 0
    for layer_a, layer_b, layer_op in layers[:-1]:
        cur_count += len(layer_a)
        prefix_sums.append(cur_count)
    return prefix_sums

def get_net_layers(model, lut_size, verbose=False):
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
                print('Skipping  layer ({}).'.format(type(layer)))
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

# if __name__ == "__main__":
#     args = get_args()
#     if args.name is None:
#         args.name = args.model

#     model_path = Path("models") / f"{args.model}.pth"
#     model = torch.load(model_path, weights_only=False)

#     out_dir = gen_sv_code(model, args.name)
#     print(f"[convert2sv] Wrote SystemVerilog to: {out_dir}")