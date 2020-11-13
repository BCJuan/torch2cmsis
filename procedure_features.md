# Series of improvements of torch2cmsis

In this document the points that will be included are described as well as waht each of the epected releases might include.
With each change a value for one or two tests is included to check the functioning of the framework (pytorch/cmsis)
## FIRST EXAMPLE

MNIST
1. Architecture -------------> add max pool --------------------> 0.9804/0.9792, 0.9778/0.9761
2. Architecture -------------> add + fc ------------------------>
3. Statistics for quant based on quantiles --------------------->

### FIRST RELEASE

1. Pass to converter
2. Organize example
3. makefile
    1. Black, isort, flake8
5. Docs and docs examples

## SECOND EXAMPLE

1. CIFAR10 q15 ------------------------------------------------>
2. Checker of activations vs quant activations (graphic)------->
3. Non square kernels ----------------------------------------->
4. depthwise convolutions ------------------------------------->

### SECOND RELEASE

1. makefile
    1. Tests
    2. CI? 

## THIRD EXAMPLE

1. Automatic code generator ----------------------------------->
    1. CNN, Max Pool, fc
    2. Headers
    3. Network
    4. Read/write
    5. main

### THIRD RELEASE

## FOURTH EXAMPLE
1. CoST GRU --------------------------------------------------->
2. Cost LSTM -------------------------------------------------->

### FOURTH RELEASE


FIFTH EXAMPLE

1. MNIST s8 quantization -------------------------------------->