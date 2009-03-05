all: nnwork oglnn

nnwork: main.c nnwork.h
	cc -o nnwork main.c -lm -g -std=c99

oglnn: oglnn.c nnwork.h
	cc -o oglnn oglnn.c -lm -g -std=c99 -lglut

clean:
	rm -f nnwork oglnn
