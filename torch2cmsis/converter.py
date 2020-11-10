import os
import subprocess
import sys
from tqdm import tqdm
from subprocess import call
import numpy as np
import torch
from torch import nn as nn
from torch import quantization

from .fully_connected_opt_weight_generation import (convert_q7_q15_weights,
                                                   convert_to_x4_q7_weights,
                                                   convert_to_x4_q15_weights)


class CMSISConverter:
    """
    This class prepares the needed files to implement a
    convolutional neural network
    in CMSIS from pytorch

    it creates two files:
        parameters.h
        weights.h 

    following the examples found at
    https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN/Examples/IAR/iar_nn_examples/NN-example-cifar10
>
    it operates following the guide
    https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/converting-a-neural-network-for-arm-cortex-m-with-cmsis-nn/single-page

    Conditions:
        Image must be squared
        Kernels must be squared
    """

    def __init__(self, model, root, weight_file_name, parameter_file_name, weight_bits=8):
        
        #TODO: defined by user should be
        self.root = root
        self.io_folder = os.path.join(self.root, "logs")
        if not os.path.exists(self.io_folder):
            os.mkdir(self.io_folder)
        
        self.model = model

        self.parameter_file_name =  os.path.join(self.root, parameter_file_name)
        self.weight_file_name =  os.path.join(self.root, weight_file_name)

        parameter_file = open(self.parameter_file_name, "w")
        parameter_file.close()
        weight_file = open(self.weight_file_name, "w")
        weight_file.close()

        # define storage for maximum buffers in CMSIS
        self.max_col_buffer = 0
        self.max_fc_buffer = 0
        
        # here we suppose an 8bit signed number in original range [0, 1]
        # which corresponds to range 7 fractional bits
        self.weight_bits = weight_bits
        self.fractional_bits = {}

        # for storing all convolution, pooling and linear params
        self.params = {}
        self.param_prefix_name = None

        # storing inputs and ouputs quantized
        self.logging = {}

    def prepare_quantization(self, loader):
        self.model.qconfig = quantization.QConfig(
            activation=quantization.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric),
            weight=quantization.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric))
        register_hooks(self.model)
        quantization.prepare(self.model, inplace=True, prehook=quantization.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric))
        
        for input, _ in loader:
            _ = self.model(input)

    def convert_model_cmsis(self):
        """
        Runs through the model searching for
        Convolutions

        Linear and pools should be implemented as conv is
        """
        # TODO: is ther a way to get the shape betwen convolutions and flatten
        # without having it embedded as function inthe model?
        count_conv = 1
        count_linear = 1
        count_pool = 1
        # TODO: the image shape befor efully connected is only known because
        # there is a function the model that gets it. Should be independent of tht function
        self.conv_linear_interface_shape = torch.tensor(self.model.get_shape())

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                self.param_prefix_name = "CONV" + str(count_conv)
                self.convert_module(module)
                self.save_params_conv(module)
                count_conv += 1

            elif isinstance(module, nn.Linear):
                self.param_prefix_name = "IP" + str(count_linear)
                self.convert_module(module)
                self.save_params_linear(module)
                count_linear += 1

            elif isinstance(module, nn.MaxPool2d):
                self.param_prefix_name = "POOL" + str(count_pool)
                self.save_params_pool(module)
                count_pool += 1
            

        self.write_shifts_n_params()
    
    def register_logging(self, sample):

        count_conv = 1
        count_linear = 1
        count_pool = 1

        _ = self.model(sample)
        # TODO: solve the following problem: actually we need the input of this method
        # to be a full batch since we are running the model inside. However, quantizing the
        #input needs only an input, so we select the 0 position of the array. What if
        # there is only on value and we are selecting a channel?
        qtensor = self.quantize_tensor(sample[0])
        qtensor.numpy().astype(np.int8).tofile(
            os.path.join(self.io_folder, 'input.raw'))

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                self.param_prefix_name = "CONV" + str(count_conv)
                self.logging[self.param_prefix_name + "_OUT"] = self.quantize_tensor(
                    module.output).numpy()
                count_conv += 1
            elif isinstance(module, nn.Linear):
                self.param_prefix_name = "IP" + str(count_linear)
                self.logging[self.param_prefix_name + "_OUT"] = self.quantize_tensor(
                    module.output).numpy()
                count_linear += 1
            elif isinstance(module, nn.MaxPool2d):
                self.param_prefix_name = "POOL" + str(count_pool)
                self.logging[self.param_prefix_name + "_OUT"] = self.quantize_tensor(
                    module.output).numpy()
                count_pool += 1

        self.write_logging()

    def convert_module(self, module):
        # call compute output bias shifts
        act_bits = self.weight_bits - 1 - compute_fractional_bits(
            module.activation_post_process.min_val,
            module.activation_post_process.max_val)
        inp_bits = self.weight_bits - 1 - compute_fractional_bits(
            module.activation_pre_process.min_val,
            module.activation_pre_process.max_val)
        # act_bits = self.weight_bits - 1 - compute_fractional_bits(
        #     module.output_min_val,
        #     module.output_max_val)
        # inp_bits = self.weight_bits - 1 - compute_fractional_bits(
        #     module.input_min_val,
        #     module.input_max_val)

        # suposes that module has two named parameters: weight and bias
        self.compute_output_bias_shifts(module.weight,
                                        module.bias, act_bits, inp_bits)
        for param in module.named_parameters():
            self.convert_conv_linear_weight_cmsis(param[0], param[1])


    def compute_output_bias_shifts(self, weight, bias, activation_bits, input_bits):

        q_weight = self.weight_bits - 1 - compute_fractional_bits(
            torch.min(weight),
            torch.max(weight)
        )
        q_bias = self.weight_bits - 1 - compute_fractional_bits(
            torch.min(bias),
            torch.max(bias)
        )
        self.fractional_bits[self.param_prefix_name + "_BIAS_LSHIFT"] = \
            input_bits + q_weight - q_bias
        self.fractional_bits[self.param_prefix_name + "_OUT_RSHIFT"] = \
            input_bits + q_weight - activation_bits
        self.fractional_bits[self.param_prefix_name + "_Q"] = q_weight
        self.fractional_bits[self.param_prefix_name + "_BIAS_Q"] = q_bias
        self.fractional_bits[self.param_prefix_name + "_INPUT_Q"] = input_bits
        self.fractional_bits[self.param_prefix_name + "_OUT_Q"] = activation_bits

    def convert_conv_linear_weight_cmsis(self, tensor_name, weight):

        if tensor_name == 'bias':
            name = self.param_prefix_name + "_BIAS"
        if tensor_name == "weight":
            name = self.param_prefix_name + "_WT"

        qweight = self.quantize_tensor(weight)

        if tensor_name == 'bias':
            self.write_weights(name, qweight.numpy().astype(np.int8))

        if tensor_name == "weight":
            if "CONV" in name:
                # torch has conv weighs (out, in, h, w) while cmsis
                # (o, h, w, i). like in tutorial for legacy
                self.write_weights(
                    name,
                    qweight.permute(0, 2, 3, 1).numpy().astype(np.int8))
            elif "IP" in name:
                original_shape = qweight.shape
                if name == "IP1":
                    trans_weight = qweight.reshape(
                        original_shape[0], 
                        *tuple(self.conv_linear_interface_shape.numpy().tolist())).permute(
                            0, 2, 3, 1).reshape(original_shape)
                else:
                    trans_weight = qweight
                weight = convert_to_x4_q7_weights(trans_weight.reshape(original_shape[0], original_shape[1], 1, 1).numpy().astype(np.int8))
                self.write_weights(name, weight)

    def quantize_tensor(self, weight):
        q_frac = self.weight_bits - 1 - compute_fractional_bits(
            torch.min(weight),
            torch.max(weight)
        )
        return torch.ceil(weight*(2**q_frac)).type(torch.int8)
    
    def save_params_conv(self, module):
        self.params[self.param_prefix_name + "_IM_CH"] = module.in_channels
        self.params[self.param_prefix_name + "_OUT_CH"] = module.out_channels

        # kernel has to be squared
        if isinstance(module.kernel_size, tuple):
            kernel = module.kernel_size[0]
        else:
            kernel = module.kernel_size
        self.params[self.param_prefix_name + "_KER_DIM"] = kernel

        if isinstance(module.padding, tuple):
            padding = module.padding[0]
        else:
            padding = module.padding
        self.params[self.param_prefix_name + "_PADDING"] = padding

        if isinstance(module.stride, tuple):
            stride = module.stride[0]
        else:
            stride = module.stride
        self.params[self.param_prefix_name + "_STRIDE"] = stride
        
        self.params[self.param_prefix_name + "_IM_DIM"] = module.input_shape[-1]
        self.params[self.param_prefix_name + "_OUT_DIM"] = module.output_shape[-1]

        col_buffer = 2*module.in_channels*kernel*kernel
        if self.max_col_buffer < col_buffer:
            self.max_col_buffer = col_buffer
            self.params["MAX_CONV_BUFFER_SIZE"] = self.max_col_buffer
        
    def save_params_pool(self, module):

        # kernel has to be squared
        if isinstance(module.kernel_size, tuple):
            kernel = module.kernel_size[0]
        else:
            kernel = module.kernel_size
        self.params[self.param_prefix_name + "_KER_DIM"] = kernel

        if isinstance(module.padding, tuple):
            padding = module.padding[0]
        else:
            padding = module.padding
        self.params[self.param_prefix_name + "_PADDING"] = padding

        if isinstance(module.stride, tuple):
            stride = module.stride[0]
        else:
            stride = module.stride
        self.params[self.param_prefix_name + "_STRIDE"] = stride
        
        self.params[self.param_prefix_name + "_IM_DIM"] = module.input_shape[-1]
        self.params[self.param_prefix_name + "_OUT_DIM"] = module.output_shape[-1]

    def save_params_linear(self, module):
        self.params[self.param_prefix_name + "_OUT"] = module.out_features
        # TODO: the image shape befor efully connected is only known because
        # there is a function the model that gets it. Should be independent of tht function
            
        self.params[self.param_prefix_name + "_DIM"] = torch.prod(
            torch.tensor(
                module.input_shape[-1:])).item()

        if self.max_fc_buffer < self.params[self.param_prefix_name + "_DIM"]:
            self.max_fc_buffer = self.params[self.param_prefix_name + "_DIM"]
            self.params["MAX_FC_BUFFER"] = self.max_fc_buffer

    def write_weights(self, name, weight):
        with open(self.weight_file_name, "a") as w_file:
            w_file.write("#define " + name + " {")
            weight.tofile(w_file, sep=',')
            w_file.write("}\n")
            w_file.write("#define " + name + "_SHAPE ")
            w_file.write(str(np.prod(weight.shape)))
            w_file.write("\n")

    def write_shifts_n_params(self):
        with open(self.parameter_file_name, "w+") as w_file:
            for i, j in self.fractional_bits.items():
                w_file.write("#define " + i + " " + str(j) + "\n")
            for i, j in self.params.items():
                w_file.write("#define " + i + " " + str(j) + "\n")

    def write_logging(self):
        for i, j in self.logging.items():
            j.tofile(os.path.join(self.io_folder, str(i).lower() + "_torch.raw"))

    def evaluate_cmsis(self, exec_path, loader):
        correct = 0
        total = 0
        for input_batch, label_batch in tqdm(loader, total=len(loader)):
            for input, label in zip(input_batch, label_batch):
                qtensor = self.quantize_tensor(input)
                qtensor.numpy().astype(np.int8).tofile(
                    os.path.join(self.io_folder, 'input.raw'))
                call(exec_path, cwd=self.root)
                #TODO: this implies that the executable produces this file
                out = np.fromfile(os.path.join(self.io_folder, "y_out.raw"), dtype=np.int8)
                pred = np.argmax(out)
                correct += (pred == label.item())
                total += 1
        print("Test accuracy for CMSIS model {}".format(correct/total))


def hook_save_params(module, input, output):
    setattr(module, "input_shape", input[0].shape)
    setattr(module, "output_shape", output[0].shape)
    setattr(module, "output", output[0])
    if module.output_max_val < torch.max(output):
        module.output_max_val = torch.max(output)
    if module.output_min_val > torch.min(output):
        module.output_min_val = torch.min(output)
    if module.input_max_val < torch.max(input[0]):
        module.input_max_val = torch.max(input[0])
    if module.input_min_val > torch.min(input[0]):
        module.input_min_val = torch.min(input[0])
    

def register_hooks(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
            module.register_buffer("input_min_val", torch.tensor(float('inf')))
            module.register_buffer("input_max_val", torch.tensor(float('-inf')))
            module.register_buffer("output_max_val", torch.tensor(float('-inf')))
            module.register_buffer("output_min_val", torch.tensor(float('inf')))
            module.register_forward_hook(hook_save_params)


def inference(model, loader):
    model.eval()

    with torch.no_grad():
        iter_loader = iter(loader)
        input, label = next(iter_loader)
        output = model(input)
    return input, label[0].item(), torch.argmax(output[0]).item()


def compute_fractional_bits(min_value, max_value):
        return int(torch.ceil(
            torch.log2(torch.max(torch.abs(max_value),
                                 torch.abs(min_value)))).item())
