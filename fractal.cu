#include <stdio.h>
#include "gpu_functions.cuh"
#include "writepgm.h"

#define IMG_WIDTH 1920
#define IMG_HEIGHT 1080
#define BLOCK_SIZEX 32
#define BLOCK_SIZEY 24

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

