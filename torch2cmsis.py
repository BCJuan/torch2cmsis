import os
import subprocess
import sys

import numpy as np
import torch
from torch import nn as nn

nnscript = os.path.abspath("CMSIS_5/CMSIS/NN/Scripts/NNFunctions")
sys.path.append(nnscript)
from fully_connected_opt_weight_generation import (convert_q7_q15_weights,
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

    def __init__(
        self,
        model,
        weight_file_name,
        parameter_file_name,
        input_file_name,
        logging_file_name,
        root="./cfiles",
        weight_bits=8,
    ):
        """
        Creates both files
        """

        # TODO: defined by user should be
        self.root = root
        self.io_folder = os.path.join(self.root, "weights")
        if not os.path.exists(self.io_folder):
            os.mkdir(self.io_folder)

        self.parameter_file_name = os.path.join(self.root, parameter_file_name)
        self.logging_file_name = os.path.join(self.root, logging_file_name)
        self.weight_file_name = os.path.join(self.root, weight_file_name)
        self.input_file_name = os.path.join(self.root, input_file_name)

        parameter_file = open(self.parameter_file_name, "w")
        parameter_file.close()
        logging_file = open(self.logging_file_name, "w")
        logging_file.close()
        weight_file = open(self.weight_file_name, "w")
        weight_file.close()
        input_file = open(self.weight_file_name, "w")
        input_file.close()

        self.weight_bits = weight_bits
        # TODO: the image shape befor efully connected is only known because
        # there is a function the model that gets it. Should be independent of tht function
        self.conv_linear_interface_shape = model.get_shape()

        # here we suppose an 8bit signed number in original range [0, 1]
        # which corresponds to range 7 fractional bits
        self.input_frac_bits = self.weight_bits - 1
        self.fractional_bits = {}

        # for storing all convolution, pooling and linear params
        self.params = {}
        self.param_prefix_name = None

        # storing inputs and ouputs quantized
        self.logging = {}

    def quantize_input(self, input):
        self.input_frac_bits = self.compute_fractional_bits_tensor(input)
        qtensor = self.quantize_tensor(input)
        self.write_io("input", qtensor.numpy().astype(np.int8))

    @staticmethod
    def compute_fractional_bits(min_value, max_value):
        return int(
            torch.ceil(
                torch.log2(torch.max(torch.abs(max_value), torch.abs(min_value)))
            ).item()
        )

    def convert_model_cmsis(self, model):
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

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.param_prefix_name = "CONV" + str(count_conv)
                self.convert_module(module)
                self.save_params_conv(module)
                count_conv += 1

            if isinstance(module, nn.Linear):
                self.param_prefix_name = "IP" + str(count_linear)
                self.convert_module(module)
                self.save_params_linear(module)
                count_linear += 1

            if isinstance(module, nn.MaxPool2d):
                self.param_prefix_name = "POOL" + str(count_pool)
                self.save_params_pool(module)
                count_pool += 1

        self.write_shifts_n_params()
        self.write_logging()

    def generate_intermediate_values(self, input, model):
        register_hooks(model)
        _ = model(input)

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

        self.logging[self.param_prefix_name + "_INPUT"] = self.quantize_tensor(
            module.input
        ).numpy()
        self.logging[self.param_prefix_name + "_OUTPUT"] = self.quantize_tensor(
            module.output
        ).numpy()

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

        self.logging[self.param_prefix_name + "_INPUT"] = self.quantize_tensor(
            module.input
        ).numpy()
        self.logging[self.param_prefix_name + "_OUTPUT"] = self.quantize_tensor(
            module.output
        ).numpy()

    def save_params_linear(self, module):
        self.params[self.param_prefix_name + "_IM_CH"] = module.in_features
        self.params[self.param_prefix_name + "_OUT"] = module.out_features
        # TODO: the image shape befor efully connected is only known because
        # there is a function the model that gets it. Should be independent of tht function
        self.params[
            self.param_prefix_name + "_IM_DIM"
        ] = self.conv_linear_interface_shape[-1].item()
        self.params[self.param_prefix_name + "_DIM"] = torch.prod(
            torch.tensor(module.input_shape[-1:])
        ).item()

        self.logging[self.param_prefix_name + "_INPUT"] = self.quantize_tensor(
            module.input
        ).numpy()
        self.logging[self.param_prefix_name + "_OUTPUT"] = self.quantize_tensor(
            module.output
        ).numpy()

    def convert_module(self, module):
        # call compute output bias shifts
        act_bits = self.compute_fractional_bits_tensor(module.output)
        inp_bits = self.compute_fractional_bits_tensor(module.input)

        # suposes that module has two named parameters: weight and bias
        self.compute_output_bias_shifts(module.weight, module.bias, act_bits, inp_bits)
        for param in module.named_parameters():
            self.convert_conv_linear_weight_cmsis(param[0], param[1])

    def compute_output_bias_shifts(self, weight, bias, activation_bits, input_bits):
        """
        Computes the shifts for the bias and activation after convolution or
        linear.
        Save the resylts in a dict for final writing as in
        https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/Examples/IAR/iar_nn_examples/NN-example-cifar10/arm_nnexamples_cifar10_weights.h
        based on the computations in
        https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/converting-a-neural-network-for-arm-cortex-m-with-cmsis-nn/single-page

        Args
        ----
        name:
            either conv or linear to save as name
        counter:
            which conv or linear layer is
        weight:
            the actual weight tensor
        bias:
            the actual bias tensor
        activation_bits:
            the fractional bits for the activation

        Returns
        -----
        """
        q_weight = self.compute_fractional_bits_tensor(weight)
        q_bias = self.compute_fractional_bits_tensor(bias)

        self.fractional_bits[self.param_prefix_name + "_BIAS_LSHIFT"] = (
            input_bits + q_weight - q_bias
        )
        self.fractional_bits[self.param_prefix_name + "_OUT_RSHIFT"] = (
            input_bits + q_weight - activation_bits
        )
        self.fractional_bits[self.param_prefix_name + "_Q"] = q_weight
        self.fractional_bits[self.param_prefix_name + "_BIAS_Q"] = q_bias
        self.fractional_bits[self.param_prefix_name + "_INPUT_Q"] = input_bits
        self.fractional_bits[self.param_prefix_name + "_OUT_Q"] = activation_bits

    def compute_fractional_bits_tensor(self, weight):
        max_value = torch.max(weight)
        min_value = torch.min(weight)

        q_int = self.__class__.compute_fractional_bits(min_value, max_value)
        return self.weight_bits - 1 - q_int

    def quantize_tensor(self, weight):
        q_frac = self.compute_fractional_bits_tensor(weight)
        return torch.ceil(weight * (2 ** q_frac)).type(torch.int8)

    def convert_conv_linear_weight_cmsis(self, tensor_name, weight):

        if tensor_name == "bias":
            name = self.param_prefix_name + "_BIAS"
        if tensor_name == "weight":
            name = self.param_prefix_name + "_WT"

        qweight = self.quantize_tensor(weight)

        if tensor_name == "bias":
            self.write_weights(name, qweight.numpy().astype(np.int8))

        if tensor_name == "weight":
            if "CONV" in name:
                # torch has conv weighs (out, in, h, w) while cmsis
                # (o, h, w, i). like in tutorial for legacy
                self.write_weights(
                    name, qweight.permute(0, 2, 3, 1).numpy().astype(np.int8)
                )
            elif "IP" in name:
                original_shape = qweight.shape
                trans_weight = (
                    qweight.reshape(
                        original_shape[0],
                        *tuple(self.conv_linear_interface_shape.numpy().tolist())
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(original_shape)
                )
                weight = convert_to_x4_q7_weights(
                    trans_weight.reshape(original_shape[0], original_shape[1], 1, 1)
                    .numpy()
                    .astype(np.int8)
                )
                self.write_weights(name, weight)

    def write_io(self, name, weight):
        weight.tofile(os.path.join(self.io_folder, "%s.raw" % (name)))
        with open(self.input_file_name, "w+") as i_file:
            i_file.write("#define INPUT {")
            weight.tofile(i_file, sep=",")
            i_file.write("}\n")

    def write_shifts_n_params(self):
        """
        Appends to the weight_file_name the shifts for the bias and the output

        Args as part of self

        Args
        ----
        self.weight_file_name:
            the name of the file

        self.fractional_bits:
            the dictionary of fractional bits for each operation and
            output of it
        """
        with open(self.parameter_file_name, "w+") as w_file:
            for i, j in self.fractional_bits.items():
                w_file.write("#define " + i + " " + str(j) + "\n")
            for i, j in self.params.items():
                w_file.write("#define " + i + " " + str(j) + "\n")

    def write_logging(self):
        with open(self.logging_file_name, "w") as w_file:
            for i, j in self.logging.items():
                w_file.write(i + " " + str(j) + "\n")

    def write_weights(self, name, weight):
        with open(self.weight_file_name, "a") as w_file:
            w_file.write("#define " + name + " {")
            weight.tofile(w_file, sep=",")
            w_file.write("}\n")
            w_file.write("#define " + name + "_SHAPE ")
            w_file.write(str(np.prod(weight.shape)))
            w_file.write("\n")

    def evaluate_cmsis(self, exec_path, loader):
        correct = 0
        total = 0
        for input_batch, label_batch in loader:
            for input, label in zip(input_batch, label_batch):
                self.quantize_input(input)
                call(os.path.join("./", exec_path), cwd=self.root)
                # TODO: this implies that the executable produces this file
                out = np.fromfile(
                    os.path.join(self.io_folder, "y_out.raw"), dtype=np.int8
                )
                pred = np.argmax(out)
                correct += pred == label.item()
                total += 1
        print("Test accuracy for CMSIS model {}".format(correct / total))


def hook_save_params(module, input, output):
    setattr(module, "input_shape", input[0].shape)
    setattr(module, "output_shape", output[0].shape)
    setattr(module, "input", input[0][0])
    setattr(module, "output", output[0])


def register_hooks(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.register_forward_hook(hook_save_params)


def inference(model, loader):
    model.eval()

    with torch.no_grad():
        iter_loader = iter(loader)
        input, label = next(iter_loader)
        output = model(input)
    return input, label[0].item(), torch.argmax(output[0]).item()
