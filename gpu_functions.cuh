
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

	if((x<width) && (y<height))
		if(i==255)
			output[x+y*width] = 0;//0 is black
		else
			output[x+y*width] = i;
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
