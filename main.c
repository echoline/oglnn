#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <strings.h>
#define INPUTS 2
#define HIDDENS 5
#define OUTPUTS 1
#include "nnwork.h"

// used in sigmoid
double lambda = 1;

// ctrl+c is later hijacked to exit the training loop
typedef void (*sighandler_t)(int);
bool flag;
void ctrl_c(int signal) {
	flag = false;
}

// a couple of wrapper functions to help me as i work on
// nnwork.h
double train(double *inputs, double goal, double rate, double lambda) {
	double goals[OUTPUTS], *outputs, ret;
	goals[0] = goal;
	outputs = nnwork_train(inputs, goals, rate, lambda);
	ret = outputs[0];
	return ret;
}
double run(double *inputs, double lambda) {
	double *outputs, ret;
	outputs = nnwork_run(inputs, lambda);
	ret = outputs[0];
	return ret;
}

// run the training loop.
// returns most recent epoch since last reset
// rate is learning constant
// max_err is the "goal" error between desired and observed outputs
// epochs is the number of epochs since last reset
// epoch_rate is how fast to change max_err during annealing.
int auto_train_selected(double rate, double *max_err, int epochs, double epoch_rate) {
	int i;
	struct timeval then, now;
	double last_epoch; // seconds since last "epoch"
	// truth table for exclusive or
	double input[4][3] = {
			{ 1.0, 0.0, 1.0 }, 
			{ 0.0, 1.0, 1.0 }, 
			{ 1.0, 1.0, 0.0 }, 
			{ 0.0, 0.0, 0.0 } };
	// hijacking time
	sighandler_t oldsig = signal(SIGINT, &ctrl_c);
	flag = true;
	while (flag) {
		for (i = 0; i < 4 && flag; i++) {
			if (gettimeofday(&then, NULL) == -1) { flag = false; continue; }
			// train while result greater than max_err
			while((pow(train(input[i], input[i][2], rate, lambda) - input[i][2], 2) > *max_err) && flag);

			if (gettimeofday(&now, NULL) == -1) { flag = false; continue; }
			last_epoch = (now.tv_sec + (now.tv_usec / 1000000.0)) - (then.tv_sec + (then.tv_usec / 1000000.0));
			// anneal the process of adjusting max_err based on speed of learning
			if (last_epoch <= .00001)
				*max_err /= epoch_rate;
			if (last_epoch > 3)
				*max_err *= epoch_rate;
			printf("%d: %lf seconds - MSE: %16.14lf\n", ++epochs, last_epoch, *max_err);
		}
		printf("press ctrl+c to end training loop\n");
	}
	signal(SIGINT, oldsig);
	return epochs;
}

// run the training loop a fixed amount of times
// rate is learning constant
void train_selected(double rate) {
	int ret = 0, i;
	char buf[16];
	// truth table for exclusive or
	double input[4][3] = {
			{ 1.0, 0.0, 1.0 }, 
			{ 0.0, 1.0, 1.0 }, 
			{ 1.0, 1.0, 0.0 }, 
			{ 0.0, 0.0, 0.0 } };
	// hijacking time
	sighandler_t oldsig = signal(SIGINT, &ctrl_c);
	flag = true;
	printf("how many training cycles do you want to run? ");
	if (fgets(buf, 15, stdin) == NULL) { printf("\n"); return; }
	i = atoi(buf);
	while (flag && (ret < i)) {
		train(input[ret%4], input[ret%4][2], rate, lambda);
		ret++;
	}
	printf("%d training cycles completed.\n", ret);
	signal(SIGINT, oldsig);
}

// run the network until user presses ctrl+d
void run_selected() {
	char buf[16];
	double test[3];
	while (1) {
		printf("press ctrl+d to end run\n");
		printf("please enter first bit: ");
		if (fgets(buf, 15, stdin) == NULL) { printf("\n"); return; }
		test[0] = atof(buf);
		printf("please enter second bit: ");
		if (fgets(buf, 15, stdin) == NULL) { printf("\n"); return; }
		test[1] = atof(buf);
		test[2] = run(test, lambda);
		printf("%f xor %f is %f\n", test[0], test[1], test[2]);
	}
}

// one axis of the image is the first input, the other axis
// is the other input.  pixel values represent network outputs
void graph_selected() {
	double fromx, tox, fromy, toy, incx, incy;
	char buf[16];
	double test[2];
	int x, y;
	double answer;
	unsigned char data[256][256];
	FILE *f;

	printf("from (1st bit): ");
	if (fgets(buf, 15, stdin) == NULL) { printf("\n"); return; }
	fromx = atof(buf);
	printf("to (1st bit): ");
	if (fgets(buf, 15, stdin) == NULL) { printf("\n"); return; }
	tox = atof(buf);
	printf("from (2nd bit): ");
	if (fgets(buf, 15, stdin) == NULL) { printf("\n"); return; }
	fromy = atof(buf);
	printf("to (2nd bit): ");
	if (fgets(buf, 15, stdin) == NULL) { printf("\n"); return; }
	toy = atof(buf);

	incx = tox - fromx / 256.0;
	incy = toy - fromy / 256.0;

	test[0] = fromx;
	for (x = 0; x < 256; x++) {
		test[1] = fromy;
		for (y = 0; y < 256; y++) {
			answer = run(test, lambda);
			// shouldn't be necessary as long as sigmoid is applied to output
			if (answer > 1.0) data[x][y] = 255;
			else if (answer < 0.0) data[x][y] = 0;
			else {
				data[x][y] = (answer * 255.0);
			}
			test[1] += incy;
		}
		test[0] += incx;
	}

	printf("save pgm as: ");
	if (fgets(buf, 15, stdin) == NULL) { printf("\n"); return; }
	buf[strcspn(buf, "\r\n")] = '\0';
	f = fopen(buf, "w");
	if (!f) {
		fprintf(stderr, "cannot save %s.\n", buf);
		return;
	}
	fputs("P2\n", f);
	fputs("# neural network outputs\n", f);
	fputs("256\n", f);
	fputs("256\n", f);
	fputs("255\n", f);
	for (x = 0; x < 256; x++)
		for (y = 0; y < 256; y++)
			fprintf(f, "%d ", data[x][y]);
	fclose(f);
}

// the weights of the network can be manually adjusted here, or
// set to NAN (0.0 / 0.0) to stop them from being calculated or used.
void damage_selected() {
	char buf[1024];
	bool leave = true;
	int i, h, o;
	i = h = o = -1;

	printf("please enter a weight id to \"damage\" (ie. i0h0): ");
	if (fgets(buf, 1023, stdin)) {
		// parse weight id
		if ((buf[0] == 'h') && (buf[2] == 'o')) {
			if ((buf[1] >= '0') && (buf[1] <= '0' + HIDDENS)) {
				h = buf[1] - '0';
				if ((buf[3] >= '0') && (buf[3] <= '0' + OUTPUTS)) {
					o = buf[3] - '0';
					leave = false;
				}
			}
		} else if ((buf[0] == 'i') && (buf[2] == 'h')) {
			if ((buf[1] >= '0') && (buf[1] <= '0' + INPUTS)) {
				i = buf[1] - '0';
				if ((buf[3] >= '0') && (buf[3] <= '0' + HIDDENS)) {
					h = buf[3] - '0';
					leave = false;
				}
			}
		}
		if (leave) {
			printf("invalid weight id.\n");
			return;
		}
		printf("please enter new value or off to deactivate: (ctrl+d to cancel) ");
		if (fgets(buf, 1023, stdin)) {
			if (!strncasecmp(buf, "off", 3)) // set to NAN
				if (i == -1)
					ho_weights[h][o] = 0.0 / 0.0;
				else
					ih_weights[i][h] = 0.0 / 0.0;
			else
				if (i == -1)
					ho_weights[h][o] = atof(buf);
				else
					ih_weights[i][h] = atof(buf);
		}
	}
}

int main() {
	// default adjustable values
	double rate = 0.25;
	double max_err = .03;
	double epoch_rate = 3;

	double output;
	int epochs = 0, i, h, o;
	bool running = true;
	char buf[1024];

	// initialize random weights
	nnwork_init(time(NULL));

	// menu
	while (running) {
		printf("(a)uto train\n(t)rain\n(r)un\n(s)ettings\n(g)raph\n(w)eights\n(d)amage network\n(q)uit\n");
		if (fgets(buf, 1023, stdin) == NULL) continue;
		switch(buf[0]) {
		case 'a':
			epochs = auto_train_selected(rate, &max_err, epochs, epoch_rate);
			break;
		case 't':
			train_selected(rate);
			break;
		case 'r':
			run_selected();
			break;
		case 'g':
			graph_selected();
			break;
		case 's':
			while (running) {
				printf("(l)ambda\n(r)ate constant\n(m)ax error\n(e)poch rate\n(s)eed network\n(b)ack\n");
				if (fgets(buf, 1023, stdin) == NULL) continue;
				switch(buf[0]) {
				case 'l':
					printf("lambda is currently: %f\nlambda: ", lambda);
					if (fgets(buf, 1023, stdin))
						lambda = atof(buf);
					break;
				case 'e':
					printf("epoch rate is currently: %f\nepoch rate: ", epoch_rate);
					if (fgets(buf, 1023, stdin))
						epoch_rate = atof(buf);
					break;
				case 'r':
					printf("learning rate is currently: %f\nlearning rate: ", rate);
					if (fgets(buf, 1023, stdin))
						rate = atof(buf);
					break;
				case 'm':
					printf("maximum error is currently: %f\nnew maximum error: ", max_err);
					if (fgets(buf, 1023, stdin))
						max_err = atof(buf);
					break;
				case 's':
					if (fgets(buf, 1023, stdin)) {
						epochs = 0;
						nnwork_init(atoi(buf));
					}
					break;
				case 'b':
					running = false;
					break;
				}
			}
			running = true;
			break;
		case 'd':
		case 'w':
			printf ("input to hidden weights:\n");
			for (i = 0; i < INPUTS; i++) {
				for (h = 0; h < HIDDENS; h++) {
					printf("w:i%dh%d %15.12lf |", i, h, ih_weights[i][h]);
				}
				printf("\n");
			}
			printf ("hidden to output weights:\n");
			for (o = 0; o < OUTPUTS; o++) {
				for (h = 0; h < HIDDENS; h++) {
					printf("w:h%do%d %15.12lf |", h, o, ho_weights[h][o]);
				}
				printf("\n");
			}
			if (buf[0] == 'w')
				break;
			damage_selected();
			break;
		case 'q':
			running = false;
			break;
		}
	}
}
