all: oglnn xor xor_fann

nnwork: main.c nnwork.h
	cc -o nnwork main.c -lm -g -std=c99

oglnn: oglnn.c nnwork.h
	cc -o oglnn oglnn.c -lm -g -std=c99 -lglut -lGLU -lGL -I/usr/X11R6/include -I/usr/local/include -L/usr/local/lib -L/usr/X11R6/lib

xor: xor.c nnwork.h
	cc -o xor xor.c -lm -g -std=c99 -lglut -lGLU -lGL -I/usr/X11R6/include -I/usr/local/include -L/usr/local/lib -L/usr/X11R6/lib

xor_fann: xor_fann.c
	cc -o xor_fann xor_fann.c -lm -g -std=c99 -lglut -lGLU -lGL -I/usr/X11R6/include -I/usr/local/include -L/usr/local/lib -L/usr/X11R6/lib -lfann

clean:
	rm -f nnwork oglnn xor xor_fann *~
