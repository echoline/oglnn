all: nnwork oglnn xor

nnwork: main.c nnwork.h
	cc -o nnwork main.c -lm -g -std=c99

oglnn: oglnn.c nnwork.h
	cc -o oglnn oglnn.c -lm -g -std=c99 -lglut -lGLU

xor: xor.c nnwork.h
	cc -o xor xor.c -lm -g -std=c99 -lglut -lGLU

clean:
	rm -f nnwork oglnn xor *~
