#include <stdio.h>
//#include "gpu_functions.cuh"
#include "writepgm.h"

#define IMG_WIDTH 1920
#define IMG_HEIGHT 1080
#define BLOCK_SIZEX 32
#define BLOCK_SIZEY 24
// Write a width by height 8-bit grayscale image into File "filename"
void write_pgm(unsigned char* data,unsigned int width,unsigned int height,char* filename)
{
	if (data == NULL) {
		printf("Provide a valid data pointer!\n");
		return;
	}
	if (filename == NULL) {
		printf("Provide a valid filename!\n");
		return;
	}
	if ( (width>4096) || (height>4096)) {
		printf("Only pictures upto 4096x4096 are supported!\n");
		return;
	}
	FILE *f=fopen(filename,"wb");
	if (f == NULL) 
	{
		printf("Opening File %s failed!\n",filename);
		return;
	}
	if (fprintf(f,"P5 %i %i 255\n",width,height) <= 0) {
		printf("Writing to file failed!\n");
		return;
	};
	int i;
	for (i=0;i<height;i++) {
		if (fwrite(&(data[i*width]),width,1,f) != 1) {
			printf("Writing of line %i to file failed!\n",i);
			return;
		}
	}
	fclose(f);
}
typedef struct{
	float real;
	float img;
} pixel_complex;

//a + b = c
__device__ void complex_add(pixel_complex *a,pixel_complex *b, pixel_complex* c);

//a = a^3;
__device__ void complex_power3(pixel_complex *a);

//return = |a|
__device__ float complex_abs(pixel_complex *a);

__global__ void fractal(unsigned char* output, int width, int height, pixel_complex lowerLeft, pixel_complex upperRight)
{
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;

	if(x < IMG_WIDTH && y < IMG_HEIGHT)
	{
		unsigned char i = 0;
		pixel_complex zn = {0,0};
		pixel_complex c;

		c.real = (float)x/(float)(width-1) *(upperRight.real - lowerLeft.real) + lowerLeft.real;
		c.img  = (float)y/(float)(height-1)*(upperRight.img - lowerLeft.img) + lowerLeft.img;

		while((complex_abs(&zn) < 1.0) && (i < 255))
		{
			complex_power3(&zn);
			complex_add(&zn, &c, &zn);
			i++;
		}

		if(i==255)
			output[x+y*width] = 0;//0 is black
		else
			output[x+y*width] = i;
	}
}


__device__ void complex_add(pixel_complex *a,pixel_complex *b, pixel_complex* c)
{
	float real = a->real+b->real;
	float img  = a->img+b->img;
	c->real = real;
	c->img = img;
}


__device__ void complex_power3(pixel_complex *a)
{
	float real = a->real*a->real*a->real-3*a->real*a->img*a->img;
	float img  = 3*a->real*a->real*a->img-a->img*a->img*a->img;
	a->real = real;
	a->img = img;
}

__device__ float complex_abs(pixel_complex *a)
{
	float abs;
	abs = sqrtf((a->real*a->real) + (a->img*a->img));
	return abs;
}
int main(int argc, char** argv) {
	unsigned char *out_local;
	unsigned char *out;

	cudaMalloc((void**)&out,sizeof(unsigned char)*IMG_WIDTH*IMG_HEIGHT);

	dim3 threads = dim3(BLOCK_SIZEX, BLOCK_SIZEY);
	dim3 grid = dim3((IMG_WIDTH/threads.x),(IMG_HEIGHT/threads.y)); 

	pixel_complex lowerLeft = {-2, -1};
	pixel_complex upperRight = {2, 1};

	fractal<<<grid,threads>>>(out,IMG_WIDTH,IMG_HEIGHT,lowerLeft,upperRight);

	out_local =(unsigned char*) malloc(sizeof(unsigned char)*IMG_WIDTH*IMG_HEIGHT);
	cudaMemcpy(out_local,out,sizeof(unsigned char)*IMG_WIDTH*IMG_HEIGHT,cudaMemcpyDeviceToHost);
	char* filename = "gpu_fractal.pgm";
	write_pgm(out_local, IMG_WIDTH, IMG_HEIGHT, filename);

}

