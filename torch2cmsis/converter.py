import os
from subprocess import call
import copy 
import numpy as np
import torch
from torch import (
    nn as nn,
    quantization
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from .fully_connected_opt_weight_generation import \
    convert_to_x4_q7_weights

# TODO:
# + Change interface detection to named modules DONE
# + Work with children modules DONE ----->!!!!!!!!THIS IMPOSES THAT NO GROUPING OF MODULES CAN BE DONE (SEQUENTIAL OR LIST) BECAUSE THEN THEY DO NOT APPEAR AS CHILDREN!!!!!!
# + Change buffering from individual buffering to global buffering: one buffer (duplicated) for input/output and another for column buffer
#   + The input/output buffer has to be the greatest of inputs output sizes 
#   + The col buffer has to be the greatest of column transformations for conv, pool and fc
#       + CONV: 2*ch_im_in*dim_kernel*dim_kernel
#       + POOL: 2*dim_im_out*ch_im_in
#       + FC: dim_vec


class CMSISConverter:
    def __init__(
            self,
            root,
            model,
            weight_file_name,
            parameter_file_name,
            weight_bits=8,
            compilation=None,
        ):

        # TODO: defined by user should be
        self.root = root
        self.io_folder = os.path.join(self.root, "logs")
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        if not os.path.exists(self.io_folder):
            os.mkdir(self.io_folder)

        self.devicy = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model = copy.deepcopy(model)
        self.model_qq = copy.deepcopy(model)
        self.parameter_file_name = os.path.join(self.root, parameter_file_name)
        self.weight_file_name = os.path.join(self.root, weight_file_name)

        parameter_file = open(self.parameter_file_name, "w")
        parameter_file.close()
        weight_file = open(self.weight_file_name, "w")
        weight_file.close()

        self.compilation = compilation
        # define storage for maximum buffers in CMSIS
        self.max_col_buffer = 0
        self.max_fc_buffer = 0

        self.weight_bits = weight_bits
        # TODO: the image shape befor efully connected is only known because
        # there is a function the model that gets it. Should be independent of tht function
        self.conv_linear_interface_shape = self.model.interface_shape

        # for storing all convolution, pooling and linear params
        self.params = {}
        self.q_params = {}
        self.connectivity = {}
        self.param_prefix_name = None

        # storing inputs and ouputs quantized
        self.logging = {}

    def convert_model(self, loader):
        self.generate_intermediate_values(loader)
        self.save_params_model()
        self.refine_model_weights(loader)
        self.refine_model_weights(loader, bias=True)
        self.refine_activations(loader)
        self.reassign_q_params_n_shifts()
        self.write_shifts_n_params()
        self.convert_weights()

    def module_name_adapter(self, name):
        return name.replace(".", "_").upper()

    def quantize_input(self, inp):
        qtensor = self.extract_q_quantize_tensor(inp).permute(1, 2, 0).numpy().astype(np.int8)
        qtensor.tofile(os.path.join(self.io_folder, "input.raw"))

    def compute_fractional_bits_tensor(self, weight):
        return (
            self.weight_bits
            - 1
            - compute_fractional_bits(torch.min(weight), torch.max(weight))
        )

    def quantize_tensor(self, weight, q_frac):
        return torch.ceil(weight * (2 ** q_frac)).type(torch.int8)

    def extract_q_quantize_tensor(self, weight):
        q_frac = self.compute_fractional_bits_tensor(weight)
        return self.quantize_tensor(weight, q_frac)

    def generate_intermediate_values(self, loader):
        self.model.qconfig = quantization.QConfig(
            activation=quantization.HistogramObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_affine),
            weight=quantization.HistogramObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_affine))
        self.model = quantization.prepare(
            self.model,
            prehook=quantization.HistogramObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_affine))
        register_hooks(self.model)
        self.model.to(self.devicy)
        for sample, _ in loader:
            sample = sample.to(self.devicy)
            _ = self.model(sample)

    def save_params_model(self):
        previous_module = "input"
        for module in self.model.named_children():
            self.param_prefix_name = module[0]
            if isinstance(module[1], nn.Conv2d):
                self.save_params_conv(module[1])
            if isinstance(module[1], nn.Linear):
                self.save_params_linear(module[1])
            if isinstance(module[1], (nn.MaxPool2d, nn.AvgPool2d)):
                self.save_params_pool(module[1])
            if isinstance(module[1], (nn.Conv2d, nn.Linear)):
                self.connectivity[self.param_prefix_name] = previous_module
                previous_module = module[0]
                self.q_params[self.param_prefix_name] = {}
                self.save_qparams_module(module[1])
                

    def return_act_inp_bits(self, module):
        act_bits = self.weight_bits - 1 - compute_fractional_bits(
            module.activation_post_process.min_val, module.activation_post_process.max_val)
        inp_bits = self.weight_bits - 1 - compute_fractional_bits(
            module.activation_pre_process.min_val, module.activation_pre_process.max_val)
        return act_bits, inp_bits

    def save_qparams_module(self, module):
        act_bits, inp_bits = self.return_act_inp_bits(module)
        self.compute_output_bias_shifts(module.weight, module.bias, act_bits, inp_bits)

    def compute_output_bias_shifts(
            self, weight, bias, activation_bits, input_bits
        ):
        q_weight = self.compute_fractional_bits_tensor(weight)
        q_bias = self.compute_fractional_bits_tensor(bias)

        self.q_params[self.param_prefix_name]["BIAS_LSHIFT"] = (
            input_bits + q_weight - q_bias
        )
        self.q_params[self.param_prefix_name]["OUT_RSHIFT"] = (
            input_bits + q_weight - activation_bits
        )
        self.q_params[self.param_prefix_name]["WEIGHT_Q"] = q_weight
        self.q_params[self.param_prefix_name]["BIAS_Q"] = q_bias
        self.q_params[self.param_prefix_name]["INPUT_Q"] = input_bits
        self.q_params[self.param_prefix_name]["OUT_Q"] = activation_bits

    def save_params_conv(self, module):
        self.params[self.param_prefix_name.upper() + "_IM_CH"] = module.in_channels
        self.params[self.param_prefix_name.upper() + "_OUT_CH"] = module.out_channels

        # kernel has to be squared
        if isinstance(module.kernel_size, tuple):
            kernel = module.kernel_size[0]
        else:
            kernel = module.kernel_size
        self.params[self.param_prefix_name.upper() + "_KER_DIM"] = kernel

        if isinstance(module.padding, tuple):
            padding = module.padding[0]
        else:
            padding = module.padding
        self.params[self.param_prefix_name.upper() + "_PADDING"] = padding

        if isinstance(module.stride, tuple):
            stride = module.stride[0]
        else:
            stride = module.stride
        self.params[self.param_prefix_name.upper() + "_STRIDE"] = stride

        self.params[self.param_prefix_name.upper() + "_IM_DIM"] = module.input_shape[-1]
        self.params[self.param_prefix_name.upper() + "_OUT_DIM"] = module.output_shape[-1]

        col_buffer = 2 * module.in_channels * kernel * kernel
        if self.max_col_buffer < col_buffer:
            self.max_col_buffer = col_buffer
            self.params["MAX_CONV_BUFFER_SIZE"] = self.max_col_buffer

    def save_params_pool(self, module):
        self.params[self.param_prefix_name.upper() + "_IM_CH"] = module.input_shape[1]

        # kernel has to be squared
        if isinstance(module.kernel_size, tuple):
            kernel = module.kernel_size[0]
        else:
            kernel = module.kernel_size
        self.params[self.param_prefix_name.upper() + "_KER_DIM"] = kernel

        if isinstance(module.padding, tuple):
            padding = module.padding[0]
        else:
            padding = module.padding
        self.params[self.param_prefix_name.upper() + "_PADDING"] = padding

        if isinstance(module.stride, tuple):
            stride = module.stride[0]
        else:
            stride = module.stride
        self.params[self.param_prefix_name.upper() + "_STRIDE"] = stride

        self.params[self.param_prefix_name.upper() + "_IM_DIM"] = module.input_shape[-1]
        self.params[self.param_prefix_name.upper() + "_OUT_DIM"] = module.output_shape[-1]

    def save_params_linear(self, module):
        self.params[self.param_prefix_name.upper() + "_OUT"] = module.out_features
        self.params[self.param_prefix_name.upper() + "_DIM"] = torch.prod(
            torch.tensor(module.input_shape[-1:])).item()

        if self.max_fc_buffer < self.params[self.param_prefix_name.upper() + "_DIM"]:
            self.max_fc_buffer = self.params[self.param_prefix_name.upper() + "_DIM"]
            self.params["MAX_FC_BUFFER"] = self.max_fc_buffer

#################################################################################################3
    def refine_model_weights(self, loader, bias=False):
        if bias:
            index_q = "BIAS_Q"
            index = ".bias"
        else:
            index_q = "WEIGHT_Q"
            index = ".weight"
        model_usage = copy.deepcopy(self.model_qq)
        model_usage = model_usage.to(self.devicy)
        best_accuracy = evaluate(model_usage, loader, self.devicy)
        for key in tqdm(self.q_params.keys(), desc="Refining " + index):
            q_ = self.q_params[key][index_q]
            for new_q in range(q_ + 1, self.weight_bits - 1):
                model_usage.state_dict()[key + index] = self.convert_saturate_deconvert(
                    model_usage.state_dict()[key + index], new_q)
                new_accuracy = evaluate(model_usage, loader, self.devicy)
                model_usage = copy.deepcopy(self.model_qq)
                model_usage = model_usage.to(self.devicy)
                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    self.q_params[key][index_q] = new_q
            self.model_qq.state_dict()[key + index] = self.convert_saturate_deconvert(
                    self.model_qq.state_dict()[key + index], self.q_params[key][index_q])              

    def convert_saturate_deconvert(self, matrix, q):
        matrix = self.quantize_tensor(matrix, q)
        matrix[matrix > 126] = 127
        matrix[matrix < -127] = -128
        return self.dequantize_tensor(matrix, q)

    def dequantize_tensor(self, matrix, q):
        return matrix.type(torch.float32)/2**q

    def refine_activations(self, loader):
        for key in tqdm(self.q_params.keys(), desc="Refining activations"):
            q_ = self.q_params[key]["OUT_Q"]
            model_usage = copy.deepcopy(self.model_qq)
            model_usage = model_usage.to(self.devicy)
            best_accuracy = self.evaluate_modules(model_usage, loader, key, q_)
            for new_q in range(q_ + 1, self.weight_bits - 1):
                new_accuracy = self.evaluate_modules(model_usage, loader, key, new_q)
                model_usage = copy.deepcopy(self.model_qq)
                model_usage = model_usage.to(self.devicy)
                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    self.q_params[key]["OUT_Q"] = new_q

    def evaluate_modules(self, model, loader, key, q_):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.devicy)
                labels = torch.squeeze(labels).to(self.devicy)
                out = inputs
                for name, module in model.named_children():
                    out = module(out)
                    if name == key:
                        out = self.convert_saturate_deconvert(out, q_)
                _, preds = torch.max(out, -1)
                total += inputs.shape[0]
                correct += (preds == labels).sum().item()
        return correct/total

    def reassign_q_params_n_shifts(self):
        for module in self.model.named_children():
            self.param_prefix_name = module[0]
            if isinstance(module[1], (nn.Conv2d, nn.Linear)):
                self.change_q_params()

    def change_q_params(self):
        if self.connectivity[self.param_prefix_name] in self.q_params.keys():
            self.q_params[self.param_prefix_name]["INPUT_Q"] = self.q_params[
                self.connectivity[self.param_prefix_name]]["OUT_Q"]

        self.q_params[self.param_prefix_name]["BIAS_LSHIFT"] = (
            self.q_params[self.param_prefix_name]["INPUT_Q"] +
            self.q_params[self.param_prefix_name]["WEIGHT_Q"] -
            self.q_params[self.param_prefix_name]["BIAS_Q"]
        )
        self.q_params[self.param_prefix_name]["OUT_RSHIFT"] = (
            self.q_params[self.param_prefix_name]["INPUT_Q"] +
            self.q_params[self.param_prefix_name]["WEIGHT_Q"] -
            self.q_params[self.param_prefix_name]["OUT_Q"]
        )

    def write_shifts_n_params(self):
        with open(self.parameter_file_name, "w+") as w_file:
            for i, j in self.params.items():
                w_file.write("#define " + i + " " + str(j) + "\n")
            for i, j in self.q_params.items():
                for k, l in j.items():
                    w_file.write("#define " + i.upper() + "_" + k + " " + str(l) + "\n")

    def convert_weights(self):
        self.model.to('cpu')
        for module in self.model.named_children():
            self.param_prefix_name = module[0]
            if isinstance(module[1], nn.Conv2d):
                for param in module[1].named_parameters():
                    self.convert_conv_weight(param[0], param[1])
            if isinstance(module[1], nn.Linear):
                for param in module[1].named_parameters():
                    self.convert_linear_weight(module[0], param[0], param[1])               

    def choose_bias_weight(self, tensor_name, weight):
        if tensor_name == "bias":
            name = self.param_prefix_name.upper() + "_BIAS"
            qweight = self.quantize_tensor(weight, self.q_params[self.param_prefix_name]["BIAS_Q"])
        if tensor_name == "weight":
            name = self.param_prefix_name.upper() + "_WT"
            qweight = self.quantize_tensor(weight, self.q_params[self.param_prefix_name]["WEIGHT_Q"])
        return name, qweight

    def convert_linear_weight(self, module_name, tensor_name, weight):
        name, qweight = self.choose_bias_weight(tensor_name, weight)
        if tensor_name == "bias":
            self.write_weights(name, qweight.numpy().astype(np.int8))
        if tensor_name == "weight":
            original_shape = qweight.shape
            if "interface" in module_name:
                trans_weight = (
                    qweight.reshape(
                        original_shape[0],
                        *tuple(
                            torch.tensor(self.conv_linear_interface_shape)
                            .numpy()
                            .tolist()
                        ),
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(original_shape)
                )
                weight = convert_to_x4_q7_weights(
                    trans_weight.reshape(
                        original_shape[0], original_shape[1], 1, 1
                    )
                    .numpy()
                    .astype(np.int8)
                )
            else:
                weight = convert_to_x4_q7_weights(
                    qweight.reshape(
                        original_shape[0], original_shape[1], 1, 1
                    )
                    .numpy()
                    .astype(np.int8)
                )
            self.write_weights(name, weight)
        
    def convert_conv_weight(self, tensor_name, weight):
        name, qweight = self.choose_bias_weight(tensor_name, weight)
        if tensor_name == "bias":
            self.write_weights(name, qweight.numpy().astype(np.int8))
        if tensor_name == "weight":
            self.write_weights(
                    name, qweight.permute(0, 2, 3, 1).numpy().astype(np.int8)
                )

    def write_weights(self, name, weight):
        with open(self.weight_file_name, "a") as w_file:
            w_file.write("#define " + name + " {")
            weight.tofile(w_file, sep=",")
            w_file.write("}\n")
            w_file.write("#define " + name + "_SHAPE ")
            w_file.write(str(np.prod(weight.shape)))
            w_file.write("\n")

    def compile(self):
        call(self.compilation, cwd=self.root, shell=True)

    def execute(self, exec_path):
        call(os.path.join("./", exec_path), cwd=self.root)
        # TODO: this implies that the executable produces this file
        return np.fromfile(os.path.join(self.io_folder, "y_out.raw"), dtype=np.int8)

    def evaluate_cmsis(self, exec_path, loader):
        correct = 0
        total = 0
        self.compile()
        for input_batch, label_batch in tqdm(loader, total=len(loader)):
            for inp, label in zip(input_batch, label_batch):
                self.quantize_input(inp)
                out = self.execute(exec_path)
                pred = np.argmax(out)
                correct += pred == label.item()
                total += 1
        print("Test accuracy for CMSIS model {}".format(correct / total))

###################################################################################################

    def write_logging(self):
        for i, j in self.logging.items():
            j.tofile(
                os.path.join(self.io_folder, str(i).lower() + "_torch.raw")
            )

    def register_logging(self):
        for module in self.model.named_children():
            if isinstance(module[1], (nn.Conv2d, nn.Linear)):
                self.param_prefix_name = module[0]
                self.logging[self.param_prefix_name + "_OUT"] = \
                    self.quantize_tensor(
                        module[1].output,
                        self.q_params[self.param_prefix_name]["OUT_Q"]).numpy()
        self.write_logging()

    def sample_inference_checker(self, exec_path, inp, draw=False):
        self.compile()
        self.quantize_input(inp[0])
        out = self.execute(exec_path)
        out_torch = self.model(inp)[0]
        self.register_logging()
        self.draw_model_comparison(draw)

    def draw_model_comparison(self, draw=False):
        for module in self.model.named_children():
            self.param_prefix_name = module[0]
            if isinstance(module[1], (nn.Conv2d, nn.Linear)):
                if draw:
                    draw_activation(
                        os.path.join(
                            self.io_folder,
                            self.param_prefix_name.lower() + "_out_torch.raw"),
                        os.path.join(
                            self.io_folder,
                            self.param_prefix_name.lower() + "_out.raw")
                    )

def hook_save_params(module, input, output):
    setattr(module, "input_shape", input[0].shape)
    setattr(module, "output_shape", output[0].shape)
    setattr(module, "input", input[0][0])
    setattr(module, "output", output[0])

def register_hooks(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d)):
            module.register_forward_hook(hook_save_params)


def compute_fractional_bits(min_value, max_value):
    return int(
        torch.ceil(
            torch.log2(torch.max(torch.abs(max_value), torch.abs(min_value)))
        ).item()
    )

def draw_activation(torch_activation_name, cmsis_activation_name):
    torch_activation = np.sort(np.fromfile(torch_activation_name, dtype=np.int8))
    cmsis_activation = np.sort(np.fromfile(cmsis_activation_name, dtype=np.int8))
    label = torch_activation_name.split("_")[0]
    plt.plot(torch_activation, label="PyTorch " + label, c='k')
    plt.plot(cmsis_activation, label="CMSIS-NN " + label, c='r')
    plt.legend()
    plt.show()

def evaluate(model, loader, device):
    model.eval().to(device)
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = torch.squeeze(labels).to(device)
            out = model(inputs)
            _, preds = torch.max(out, -1)
            total += inputs.shape[0]
            correct += (preds == labels).sum().item()
    return correct/total