#include <GL/glut.h>
#define __USE_XOPEN_EXTENDED
#include <unistd.h>
#include <time.h>
#include <string.h>
#define INPUTS 256
#define HIDDENS 256
#define OUTPUTS 10
#include "nnwork.h"
#define SCREEN_WIDTH 500
#define SCREEN_HEIGHT 500
#define BUFSIZE 512

// RGBA colors
float neuron[] = {0.0f, 0.0f, 0.0f, 1.0f};
float negative[] = {0.0f, 0.7f, 0.0f, 1.0f};
float positive[] = {0.0f, 0.0f, 0.7f, 1.0f};

double lambda = 1, rate = 0.25;
int speed = 1;
int counter = 0;
int mx, my;
float angleX = 10, angleY = 10;
double input[256], output[10];
double depth = -175.0f;
FILE *train;

void
draw_text(GLint x, GLint y, char* s, GLfloat r, GLfloat g, GLfloat b)
{
	int lines;
	char* p;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, glutGet(GLUT_WINDOW_WIDTH), 
		0.0, glutGet(GLUT_WINDOW_HEIGHT), -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(r,g,b);
	glMaterialfv(GL_FRONT, GL_EMISSION, positive);
	glRasterPos2i(x, y);
	for(p = s, lines = 0; *p; p++) {
		if (*p == '\n') {
			lines++;
			glRasterPos2i(x, y+(lines*18));
		} else
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
	}
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}

// process the hits that happened during selection
void process_hits(GLint hits, GLuint buffer[])
{
	int y = 0, i, h, o;
	if(hits > 0)//Make sure there is at least one hit
	{
		//The 4th spot, array index 3, is the top of the stack, 
		//which holds the id of the last item that was drawn to the place we clicked on
		y = buffer[3];
		if (y >= INPUTS*HIDDENS) {
			h = (y - INPUTS*HIDDENS) % HIDDENS;
			o = (y - INPUTS*HIDDENS) / HIDDENS;
			ho_weights[h][o] = 0.0/0.0;
		} else {
			i = y / HIDDENS;
			h = y % HIDDENS;
			ih_weights[i][h] = 0.0/0.0;
		}
	}      
}

void draw_net(int mode, double *input, double *output) {
	int i, h, o, id = 0, rt;
	// input to hidden
	glPushMatrix();
	glTranslatef(0,0,depth);
	glRotatef(angleX, 0, 1, 0);
	glRotatef(-angleY, 1, 0, 0);
	
	for (i = 0; i < INPUTS; i++) {
		for (h = 0; h < HIDDENS; h++) {
			if (!isnormal(ih_weights[i][h]))
				continue;
			else if (ih_weights[i][h] < 0)
				glMaterialfv(GL_FRONT, GL_EMISSION, negative);
			else
				glMaterialfv(GL_FRONT, GL_EMISSION, positive);
			glLineWidth(abs(ih_weights[i][h]));
			if (mode == GL_SELECT)
				glPushName(i*HIDDENS+h);
			glBegin(GL_LINES);
				rt = sqrt(INPUTS);
				glVertex3f(((i%rt)*10.0f)-((rt-1)*5.0f), 10.0f, 10.0f*(i/rt)-(rt-1)*5.0f);
				rt = sqrt(HIDDENS);
				glVertex3f(((h%rt)*10.0f)-((rt-1)*5.0f), 0.0f, 10.0f*(h/rt)-(rt-1)*5.0f);
			glEnd();
			if (mode == GL_SELECT)
				glPopName();
		}
	}
	// hidden to output
	for (o = 0; o < OUTPUTS; o++) {
		for (h = 0; h < HIDDENS; h++) {
			if (!isnormal(ho_weights[h][o]))
				continue;
			else if (ho_weights[h][o] < 0)
				glMaterialfv(GL_FRONT, GL_EMISSION, negative);
			else
				glMaterialfv(GL_FRONT, GL_EMISSION, positive);
			glLineWidth(abs(ho_weights[h][o]));
			if (mode == GL_SELECT)
				glPushName(INPUTS*HIDDENS+o*HIDDENS+h);
			glBegin(GL_LINES);
				rt = sqrt(OUTPUTS);
				glVertex3f(((o%rt)*10.0f)-((rt-1)*5.0f), -10.0f, 10.0f*(o/rt)-(rt-1)*5.0f);
				rt = sqrt(HIDDENS);
				glVertex3f(((h%rt)*10.0f)-((rt-1)*5.0f), 0.0f, 10.0f*(h/rt)-(rt-1)*5.0f);
			glEnd();
			if (mode == GL_SELECT)
				glPopName();
		}
	}

	// input nodes
	rt = sqrt(INPUTS);
	for (i = 0; i < INPUTS; i++) {
		neuron[0] = (-input[i] + 1.0) / 2.0;
		glMaterialfv(GL_FRONT, GL_EMISSION, neuron);
		glTranslatef(((i%rt)*10.0f)-((rt-1)*5.0f), 10.0f, 10.0f*(i/rt)-(rt-1)*5.0f);
		glutSolidSphere(1,20,20);
		glTranslatef(((i%rt)*-10.0f)+((rt-1)*5.0f), -10.0f, -10.0f*(i/rt)+(rt-1)*5.0f);
	}

	// hidden nodes
	rt = sqrt(HIDDENS);
	for (h = 0; h < HIDDENS; h++) {
		neuron[0] = hidden_outputs[h];
		glMaterialfv(GL_FRONT, GL_EMISSION, neuron);
		glTranslatef(((h%rt)*10.0f)-((rt-1)*5.0f), 0.0f, 10.0f*(h/rt)-(rt-1)*5.0f);
		glutSolidSphere(1,20,20);
		glTranslatef(((h%rt)*-10.0f)+((rt-1)*5.0f), 0.0f, -10.0f*(h/rt)+(rt-1)*5.0f);
	}

	// output nodes
	rt = sqrt(OUTPUTS);
	for (o = 0; o < OUTPUTS; o++) {
		neuron[0] = output[o];
		glMaterialfv(GL_FRONT, GL_EMISSION, neuron);
		glTranslatef(((o%rt)*10.0f)-((rt-1)*5.0f), -10.0f, 10.0f*(o/rt)-(rt-1)*5.0f);
		glutSolidSphere(1,20,20);
		glTranslatef(((o%rt)*-10.0f)+((rt-1)*5.0f), 10.0f, -10.0f*(o/rt)+(rt-1)*5.0f);
	}
	glPopMatrix();
}

void kbd(unsigned char key, int x, int y)
{
	if (key == 's') {
		nnwork_init(time(NULL));
		counter = 0;
	}
	if (key == '1') {
		if (speed > 1) speed--;
	}
	if (key == '2') {
		if (speed < 100) speed++;
	}
	if (key == '3') {
		if (lambda > 0.01) lambda *= .95;
	}
	if (key == '4') {
		if (lambda < 20) lambda /= .95;
	}
	if (key == '5') {
		if (rate > 0.001) rate -= 0.001;
	}
	if (key == '6') {
		if (rate < 1) rate += 0.001;
	}
}

void mbutton(int button, int state, int x, int y)
{
	GLuint selectBuf[BUFSIZE];//create the selection buffer
	GLint hits;
	GLint viewport[4];//create a viewport

	//check for a left mouse click
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		glGetIntegerv (GL_VIEWPORT, viewport);//set the viewport
		glSelectBuffer (BUFSIZE, selectBuf);//set the select buffer
		(void) glRenderMode (GL_SELECT);//put OpenGL in select mode
		glInitNames();//init the name stack
		glPushName(0);//push a fake id on the stack to prevent load error
		glPopName();  // get the zero off the stack.		
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
			glLoadIdentity(); 
			//setup a pick matrix
			gluPickMatrix((GLdouble)x, (GLdouble)(viewport[3]-y), 
					4.0, 4.0, viewport);
			gluPerspective(45.0, SCREEN_WIDTH/SCREEN_HEIGHT, 0.1, 1000.0);
			draw_net(GL_SELECT, input, output);// Draw to the pick matrix instead of our normal one
			glMatrixMode(GL_PROJECTION);
		glPopMatrix ();
		glFlush ();
		hits = glRenderMode (GL_RENDER);//count the hits
		process_hits (hits, selectBuf);//check for object selection
		glutPostRedisplay();
	}
}

void mmove(int x, int y) {
	angleX = (x - mx) * (180.0f/SCREEN_WIDTH);
	angleY = (y - my) * (180.0f/SCREEN_HEIGHT);
}

void display(void) {
	char buf[1024];
	double *results;
	double error = 0;
	int i, h, o;

	h = 0;
	for(i = 0; !feof(train); i++) {
		buf[i] = fgetc(train);
		if (buf[i] == ' ') {
			input[h++] = atof(buf);
			if (h > 255) {
				buf[0] = fgetc(train);
				fgetc(train);
				buf[1] = '\0';
				h = atoi(buf);
				for (i = 0; i < 10; i++)
					output[i] = (i == h) ? 1.0 : 0.0;
				break;
			}
			i = 0;
		}
	}
	if (feof(train)) {
		rewind(train);
	}

	results = nnwork_train(input, output, rate, lambda);
	for (o = 0; o < sizeof(results); o++)
		error += pow(output[o] - results[o], 2);

	counter++;
	if (counter % speed) {
		free(results);
		return;
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	sprintf(buf, "Epochs: %d\nError: %20.18lf\nPress s to seed network\n"
			"5 and 6 adjust learning rate: %f\n"
			"3 and 4 adjust lambda: %f\n"
			"1 and 2 adjust speed: %d\n",
			counter, error, rate, lambda, speed);
	draw_text(0, 0, buf, 0, 1, 0);
	draw_net(GL_RENDER, input, results);
	free(results);

	glFlush();
	glutSwapBuffers();
	glutPostRedisplay();
}

void resize(int w, int h)
{
        glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
        glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
        gluPerspective(45.0, SCREEN_WIDTH / SCREEN_HEIGHT, 0.01, 1000.0);
        glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutWarpPointer(SCREEN_WIDTH, SCREEN_HEIGHT);
}

int main(int argc, char **argv) {
	GLfloat mat_specular[] = { 0.0, 0.0, 1.0, 0.5 };
	GLfloat mat_shininess[] = { 75.0 };
	GLfloat mat_diffuse[] = { 0.0, 0.0, 1.0, 0.5 };
	GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[] = { 0.5, 0.5, 0.5, 1.0 };
	GLfloat light_specular[] = { 0.5, 0.5, 0.5, 1.0 };
	GLfloat light_position[] = { 0.0, 6.0, -70.0, 1.0 };

	if (argc < 2) {
		printf("please specify a training file\n");
		return -1;
	}
	train = fopen(argv[1], "r");
	if (!train) {
		printf("unable to open training file\n");
		return -1;
	}

	glutInit(&argc, argv);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow("nnwork");
	glutReshapeFunc(resize);
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutMouseFunc(mbutton);
	glutMotionFunc(mmove);
	glutKeyboardFunc(kbd);

	// antialiasing
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT,GL_NICEST);
	
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

	nnwork_init(time(NULL));

	glutMainLoop();
}
