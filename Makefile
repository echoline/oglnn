all: oglnn xor

nnwork: main.c nnwork.h
	cc -o nnwork main.c -lm -g -std=c99

oglnn: oglnn.c nnwork.h
	cc -o oglnn oglnn.c -lm -g -std=c99 -lglut -lGLU -lGL -I/usr/X11R6/include -I/usr/local/include -L/usr/local/lib -L/usr/X11R6/lib

xor: xor.c nnwork.h
	cc -o xor xor.c -lm -g -std=c99 -lglut -lGLU -lGL -I/usr/X11R6/include -I/usr/local/include -L/usr/local/lib -L/usr/X11R6/lib

clean:
	rm -f nnwork oglnn xor *~
