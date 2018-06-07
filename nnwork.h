#ifndef NNWORK
#define NNWORK
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef INPUTS
	#define INPUTS 2
#endif
#ifndef HIDDENS
	#define HIDDENS 5
#endif
#ifndef OUTPUTS
	#define OUTPUTS 1
#endif

// one node for each pair.
// set to a non-normal double value to disable
// ie inf, nan
double ih_weights[INPUTS][HIDDENS];
double ho_weights[HIDDENS][OUTPUTS];
// outputs of hidden layer
double hidden_outputs[HIDDENS];
// typedef for logistic function so that it can be swapped
// with a different function.  train() will only work with
// the default
typedef double (*sigmoid_func_t)(double,double);

// this is the default logistic function.
// if this is modified, train() must also be modified
// because it uses this function's derivative
// can be overridden by setting sigmoid_func
double nnwork_sigmoid(double input, double lambda) {
	return (1.0 / (1.0 + exp(-input*lambda)));
//	return input*lambda;
//	return (input > 0.0? input*lambda: 0.0);
}
double nnwork_relu(double input, double lambda) {
	//return (1.0 / (1.0 + exp(-input*lambda)));
	return (input > 0.0? input*lambda: 0.01*input*lambda);
}
double nnwork_tanh(double input, double lambda) {
	return tanh (input*lambda);
}
sigmoid_func_t hidden_func = nnwork_relu;
sigmoid_func_t output_func = nnwork_sigmoid;

// initialize network with a given seed for random
// weight generation
void nnwork_init(unsigned long seed) {
	int i, h, o;

	srand(seed);
	for (i = 0; i < INPUTS; i++)
		for (h = 0; h < HIDDENS; h++)
			ih_weights[i][h] = 0.5 - (rand() / (double)(RAND_MAX));
	for (h = 0; h < HIDDENS; h++)
		for (o = 0; o < OUTPUTS; o++)
			ho_weights[h][o] = 0.5 - (rand() / (double)(RAND_MAX));
}

// returns the output nodes
// input is array of size INPUTS
// lambda is used in sigmoid function
double *nnwork_run(double *input, double lambda) {
	double sum;
	double *ret = (double*)malloc(sizeof(double) * OUTPUTS);
	int i, h, o;

	for (h = 0; h < HIDDENS; h++) {
		sum = 0;
		for (i = 0; i < INPUTS; i++) {
			if (isnormal(ih_weights[i][h]))
				sum += ih_weights[i][h] * input[i];
		}
		hidden_outputs[h] = hidden_func(sum, lambda);
	}

	for (o = 0; o < OUTPUTS; o++) {
		sum = 0;
		for (h = 0; h < HIDDENS; h++)
			if (isnormal(ho_weights[h][o]))
				sum += ho_weights[h][o] * hidden_outputs[h];
		ret[o] = output_func(sum, lambda);
	}
	return ret;
}

// returns the "input" nodes
// input is array of size OUTPUTS
// lambda is used in sigmoid function
/***
double *nnwork_run_backwards(double *output, double lambda) {
	double sum;
	double *ret = malloc(sizeof(double) * INPUTS);
	int i, h, o;

	for (o = 0; o < OUTPUTS; o++) {
		sum = 0;
		for (h = 0; h < HIDDENS; h++)
			if (isnormal(ho_weights[h][o]))
				sum += ho_weights[h][o] * output[o];
		hidden_outputs[h] = hidden_func(sum, lambda);
	}

	for (h = 0; h < HIDDENS; h++) {
		sum = 0;
		for (i = 0; i < INPUTS; i++) {
			if (isnormal(ih_weights[i][h]))
				sum += ih_weights[i][h] * hidden_outputs[h];
		}
		ret[i] = sigmoid_func(sum, lambda);
	}
	return ret;
}
***/

// returns the output of run()
// input is array of size INPUTS
// goal is desired output value
// rate is learning constant
// lambda is used in sigmoid function
double *nnwork_train(double *input, double *goal, double rate, double lambda) {
	double *output;
	double deltas[OUTPUTS];
	double ho_delta[HIDDENS][OUTPUTS];
	double sum;
	int i, h, o;

	// run the network so we can compute the error
	output = nnwork_run(input, lambda);

	// compute the delta values for each of the outputs
	for (o = 0; o < OUTPUTS; o++) {
		deltas[o] = (goal[o] - output[o]) * output[o] * (1.0 - output[o]);
//		deltas[o] = (goal[o] - output[o]) * lambda;
//		deltas[o] = (goal[o] - output[o]) * (1.0 - pow(tanh(output[0]), 2.0));
	}

	// hidden to output change
	for (o = 0; o < OUTPUTS; o++)
		for (h = 0; h < HIDDENS; h++)
			//  if the weight is active, adjust it
			if (isnormal(ho_weights[h][o]))
				ho_delta[h][o] = rate * deltas[o] * hidden_outputs[h];

	// input to hidden change and adjustment
	for (h = 0; h < HIDDENS; h++)
		for (i = 0; i < INPUTS; i++) {
			// if the weight is active,
			if (isnormal(ih_weights[i][h])) {
				// for each of the output nodes,
				sum = 0;
				for (o = 0; o < OUTPUTS; o++)
					// if the hidden to output weight is active
					if (isnormal(ho_weights[h][o]))
						// apply its contribution to the error
						sum += deltas[o] * ho_weights[h][o];
				// adjust the weight
				//ih_weights[i][h] += rate * (1.0 - pow(tanh(hidden_outputs[h]), 2.0)) * sum * input[i]; 
				//ih_weights[i][h] += rate * hidden_outputs[h] * (1.0 - hidden_outputs[h]) * sum * input[i];
				ih_weights[i][h] += rate * (hidden_outputs[h] > 0.0? 1.0: 0.01) * lambda * sum * input[i];
			}
		}

	// adjust the hidden to output weights
	for (o = 0; o < OUTPUTS; o++)
		for (h = 0; h < HIDDENS; h++)
			if (isnormal(ho_weights[h][o]))
				ho_weights[h][o] += ho_delta[h][o];

	// return the result of run()
	return output;
}

#endif
