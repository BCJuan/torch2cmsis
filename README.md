# torch2cmsis

Library for converting a neural network developed in Pytorch to CMSIS Legacy API. 

The idea is simple:

1. Build your network in PyTorch.
2. Add the `CMSISConverter` and run the steps

```
cm_converter = CMSISConverter(<cfiles_folder>, <pytorch_model>, <header file for weights>, <header_file_name_parameters>, <bits>, <compilation_string>
cm_converter.generate_intermediate_values(<validation_dataloader>)
cm_converter.convert_model_cmsis()
```

3. Now you have both header files with all the weights and parameters necessary for running the model with CMSIS
4. Once you have your model built with CMSIS, you can run: `cm_converter.evaluate_cmsis(<execution_file>, <test_dataloader>)` to check the accuracy.

**NOTE**: currently the automatic code generation is under development so you still have to create your network manually.
**IMPORTANT**: read te section Main Caveats, because there are certain requirements to use this package

## Installation

Just `pip install torch2cmsis`

## Examples

In the folder examples you are going to find different explained examples on how to use `torch2cmsis` at different levels of configuration.

Clone the repo recursively to include CMSIS: `git clone --recursive https://github.com/BCJuan/torch2cmsis.git`

+ MNIST Example:
    + Shows how to obtain quantized weights and parameters as header files ready to use for CMSIS models
    + How to test the CMSIS model 
    + How to obtain basic logs to manually check activations

## Development

Support matrix for differen architecture components and features.

+ [ ] Layers:
    + [x] Conv2d 
    + [x] Linear
    + [x] MaxPool2d
    + [ ] AvgPool2d
    + [ ] DepthwiseConv2d
+ [ ] Quantization
    + [x] q7
    + [ ] q15
+ Other matters:
    + [ ] Non square matrices
    + [ ] Concatenation
    + [ ] Element Wise Addition
    + [ ] Element Wise Product
+ Automatic code generation
    + ...
+ Adaptation to s8 scheme
    + ...

## Main caveats for usage (**IMPORTANT**)

+ Model needs a method which delivers the size of the interface between convolutions and the first fully connected layer. We need to solve this to make it automatic.
    + See the definition of `self.conv_linear_interface_shape` in `CMSISConverter`
+ In the CMSIS network definition, save the output of your prediction (vector level, not class) as `y_out.raw`.
    + You can see an example at `examples/mnist/cfiles7main.c`, where `save("logs/y_out.raw", y_out, sizeof(y_out));`
+ The first fully connected layer encountered inside the model is considered to be the interface between the convolution layers and fully connected ones.
    + See lines 283, 68 with the definition of `self.conv_linear_interface_shape`
    + Clever way to distinguish this than choosing the layer regarding its number: if `name== IP1`


## Contributing

There is a lot of work to do (as detailed in the Development section), so if you want to contribute do not doubt to reach me.



