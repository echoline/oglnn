#ifndef NNWORK_H
#define NNWORK_H
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

// this is the logistic function.
// if this is modified, train() must also be modified
// because it uses this function's derivative
// can be overridden by setting sigmoid_func
double nnwork_sigmoid(double input, double lambda) {
	return (1.0 / (1.0 + exp(-input*lambda)));
}
sigmoid_func_t sigmoid_func = nnwork_sigmoid;

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
	double *ret = malloc(sizeof(double) * OUTPUTS);
	int i, h, o;

	for (h = 0; h < HIDDENS; h++) {
		sum = 0;
		for (i = 0; i < INPUTS; i++) {
			if (isnormal(ih_weights[i][h]))
				sum += ih_weights[i][h] * input[i];
		}
		hidden_outputs[h] = sigmoid_func(sum, lambda);
	}

	for (o = 0; o < OUTPUTS; o++) {
		sum = 0;
		for (h = 0; h < HIDDENS; h++)
			if (isnormal(ho_weights[h][o]))
				sum += ho_weights[h][o] * hidden_outputs[h];
		ret[o] = sigmoid_func(sum, lambda);
	}
	return ret;
}

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

	output = nnwork_run(input, lambda);

	// page 468 {
	for (o = 0; o < OUTPUTS; o++)
		deltas[o] = (goal[o] - output[o]) * output[o] * (1.0 - output[o]);

	for (o = 0; o < OUTPUTS; o++)
		for (h = 0; h < HIDDENS; h++)
			if (isnormal(ho_weights[h][o]))
				ho_delta[h][o] = rate * deltas[o] * hidden_outputs[h];

	for (h = 0; h < HIDDENS; h++)
		for (i = 0; i < INPUTS; i++) {
			sum = 0;
			for (o = 0; o < OUTPUTS; o++)
				if (isnormal(ho_weights[h][o]))
					sum += deltas[o] * ho_weights[h][o];
			if (isnormal(ih_weights[i][h]))
				ih_weights[i][h] += rate * hidden_outputs[h] * (1.0 - hidden_outputs[h]) * sum * input[i];
		}

	for (o = 0; o < OUTPUTS; o++)
		for (h = 0; h < HIDDENS; h++)
			if (isnormal(ho_weights[h][o]))
				ho_weights[h][o] += ho_delta[h][o];
	// }

	return output;
}

#endif
