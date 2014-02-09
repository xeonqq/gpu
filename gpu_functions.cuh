#define ABS(x) ( (x) < 0 ? -(x) : (x) )

__device__ unsigned char get_pixel(unsigned char* frame, int x, int y, unsigned width, unsigned height);
__device__ unsigned calculate_sad(unsigned char* a,unsigned char* b,  int ax, int ay, int bx, int by, unsigned width, unsigned height);

__global__ void  motion_search(unsigned char* a,unsigned char* b, unsigned width, unsigned height, int* vx, int* vy)
{

	int j = blockIdx.y*blockDim.y+threadIdx.y;
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	int blocks_x = width/16;
	int blocks_y = height/16;

	int s,t;
	int best_diff=16*16*256;			// This is larger than the largest possible absolute difference between two blocks
	int best_x,best_y=0;

	if((i < blocks_y) && (j < blocks_x) )
	{
		for (s=-15 ; s<16 ; s++)		// Search through a -15 to 15 neighborhood
			for (t=-15 ; t<16 ; t++)
			{
				int sad=calculate_sad(a,b,j*16,i*16,j*16+t,i*16+s,  width, height);	// Calculate difference between block from first image and second image
				// Second image block shifted with (s,t)
				if (sad < best_diff)			// If we found a better match then store it
				{
					best_x = t;
					best_y = s;
					best_diff = sad;
				}
			}
		//		   printf("%i %i %f\n",best_x,best_y,best_diff/256.0f);  
		vx[j+i*blocks_x] = best_x;			// Store result
		vy[j+i*blocks_x] = best_y;
	}
}

// Gets a pixel from the Luma plane of the image, takes care of boundaries
__device__ unsigned char get_pixel(unsigned char* frame, int x, int y, unsigned width, unsigned height) 
{
	if (x >= width) x=width-1;
	if (x < 0) x=0;
	if (y >= height) y=height-1;
	if (y < 0) y=0;
	return frame[x+y*width];
}

// This calculates the sum of absolute differences between two image blocks
// unsigned char*a 		Pointer to first image
// unsigned char*b 		Pointer to second image
// int ax,int ay			X and Y Position of the block in first image
// int bx,int by			X and Y Position of the block in second image
// int width			   Width of images
// int height			   Height of images
__device__ unsigned calculate_sad(unsigned char* a,unsigned char* b,  int ax, int ay, int bx, int by, unsigned width, unsigned height)
{
	int sum=0;
	int i,j;
	for (i=0; i < 16; i++)
		for (j=0; j < 16; j++)  
			sum += ABS( get_pixel(b,ax+j,ay+i,width,height) - get_pixel(a,bx+j,by+i,width,height) );
	return sum;
}

