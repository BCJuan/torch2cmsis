# torch2cmsis

Library for converting a neural network developed in Pytorch to CMSIS Legacy API. 

## Main problems

+ Model needs a method which delivers the size of the interface between convolutions and the first fully connected layer. We need to solve this to make it automatic.
    + See lines 283, 68 with the definition of `self.conv_linear_interface_shape`
+ The first fully connected layer encountered inside the model is considered to be the interface between the convolution layers and fully connected ones.
    + See lines 283, 68 with the definition of `self.conv_linear_interface_shape`
    + Clever way to distinguish this than choosing the layer regarding its number: if `name== IP1`
+ Max  pool still not implementedx



## Development

+ [ ] Layers:
    + [x] Conv2d 
    + [x] Dense
    + [ ] MaxPool2d
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

## Contributing

There is a lot of work to do (as detailed in the Development section), so if you want to contribute do not doubt to reach me.



