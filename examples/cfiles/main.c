#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h> 
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "parameters.h"
#include "weights.h"

/*
Example of code generated by hand for an square input image,
using only convolutions and fully connected layers and 
8 bit quantization of CMSIS legacy
*/

q7_t conv1_out[CONV1_OUT_CH*CONV1_OUT_DIM*CONV1_OUT_DIM];
q7_t conv2_out[CONV2_OUT_CH*CONV2_OUT_DIM*CONV1_OUT_DIM];
q7_t conv3_out[CONV3_OUT_CH*CONV3_OUT_DIM*CONV3_OUT_DIM];
q7_t fc1_out[IP1_OUT];
q7_t fc2_out[IP2_OUT];
q7_t y_out[IP2_OUT];

q7_t conv1_w[CONV1_WT_SHAPE] = CONV1_WT;
q7_t conv1_b[CONV1_BIAS_SHAPE] = CONV1_BIAS;
q7_t conv2_w[CONV2_WT_SHAPE] =  CONV2_WT;
q7_t conv2_b[CONV2_BIAS_SHAPE] = CONV2_BIAS;
q7_t conv3_w[CONV3_WT_SHAPE] = CONV3_WT;
q7_t conv3_b[CONV3_BIAS_SHAPE] = CONV3_BIAS;
q7_t fc1_w[IP1_WT_SHAPE] = IP1_WT;
q7_t fc1_b[IP1_BIAS_SHAPE] = IP1_BIAS;
q7_t fc1_w[IP2_WT_SHAPE] = IP2_WT;
q7_t fc1_b[IP2_BIAS_SHAPE] = IP2_BIAS;

q7_t col_buffer[13*13*2*8];
q7_t scratch_buffer[13 * 13 * 9 * 8];

typedef struct {
	q7_t* input_values;
	size_t size;
} GlobalInput;

q7_t* load(const char* file)
{
	size_t sz;
	q7_t* in;
	FILE* fp = fopen(file,"rb");
	assert(fp);
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	in = malloc(sz);
	fread(in, 1, sz, fp);
	fclose(fp);
	return in;
}


GlobalInput load_n_size(const char* file)
{
	GlobalInput inputted;
	FILE* fp = fopen(file,"rb");
	assert(fp);
	fseek(fp, 0, SEEK_END);
	inputted.size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	inputted.input_values = malloc(inputted.size);
	fread(inputted.input_values, 1, inputted.size, fp);
	fclose(fp);
	return inputted;
}

void save(const char* file, q7_t* out, size_t sz)
{
	FILE* fp = fopen(file,"wb");
	fwrite(out, 1, sz, fp);
	fclose(fp);
}

uint32_t network(q7_t* input)
{
	arm_convolve_HWC_q7_basic(input, CONV1_IM_DIM, CONV1_IM_CH, conv1_w, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
						  CONV1_STRIDE, conv1_b, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, conv1_out, CONV1_OUT_DIM,
						  (q15_t *) col_buffer, NULL);
	save("weights/conv1_out.raw", conv1_out, sizeof(conv1_out));

    // first relu
    arm_relu_q7(conv1_out, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);
	save("weights/relu1_out.raw", conv1_out, sizeof(conv1_out));

    // second conv
    arm_convolve_HWC_q7_basic(conv1_out, CONV2_IM_DIM, CONV2_IM_CH, conv2_w, CONV2_OUT_CH, CONV2_KER_DIM,
						  CONV2_PADDING, CONV2_STRIDE, conv2_b, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, conv2_out,
						  CONV2_OUT_DIM, (q15_t *) col_buffer, NULL);
	save("weights/conv2_out.raw", conv2_out, sizeof(conv2_out));

    // second relu
	arm_relu_q7(conv2_out, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);
	save("weights/relu2_out.raw", conv2_out, sizeof(conv2_out));

    // first fc
	arm_fully_connected_q7_opt(conv2_out, fc1_w, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, fc1_b,
						  fc1_out, (q15_t *) scratch_buffer);
	save("weights/fc1_out.raw", fc1_out, sizeof(fc1_out));

    // softmax
    arm_softmax_q7(fc1_out, IP1_OUT, y_out);
	save("weights/y_out.raw", y_out, sizeof(y_out));

	uint32_t index[1];
	q7_t result[1];
	uint32_t blockSize = sizeof(y_out);
	// for (int i = 0; i < IP1_OUT; i++)
    // {
    //     printf("%i: %i\n", i, y_out[i]);
    // }

	arm_max_q7(y_out, blockSize, result, index);
	printf("Classified class %i\n", index[0]);

	return index[0];
}