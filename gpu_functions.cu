#include "motion.h"
#define ABS(x) ( (x) < 0 ? -(x) : (x) )

__device__ unsigned char get_pixel(unsigned char* frame, int x, int y, unsigned width, unsigned height);
__device__ unsigned calculate_sad(unsigned char* a,uchar4* b,  int ax, int ay, int bx, int by, unsigned width, unsigned height);


__global__ void  motion_search(unsigned char* a,unsigned char* b, unsigned int width, unsigned int height, int* vx, int* vy)
{
	int i = blockIdx.y*blockDim.y+threadIdx.y; // Block in Y axis
	int j = blockIdx.x*blockDim.x+threadIdx.x; // Block in X axis


	int s,t;
	//int best_diff=16*16*256;			// This is larger than the largest possible absolute difference between two blocks
	int k;
	int partial_sum = 0;
	int final_sum = 0;
	//unsigned char Bs[256];

	//__shared__ uchar4 Bs[BLOCK_SIZEX * BLOCK_SIZEY];
	
	//__shared__ uchar As[BLOCK_SIZEY * (BLOCK_SIZEX + (4*2))]; // Need 2 more tiles than the blockdim.x

	__shared__ int SUMs[BLOCK_SIZEX * BLOCK_SIZEY];
	__shared__ int SUM_ROWs[BLOCK_SIZEY*BLOCK_SIZEX/4];
	__shared__ int FINAL_SUMs[BLOCK_SIZEX/4];
	__shared__ int BEST_Xs[BLOCK_SIZEX/4];
	__shared__ int BEST_Ys[BLOCK_SIZEX/4];
	__shared__ int BEST_DIFFs[BLOCK_SIZEX/4];

	uchar4 B;
		
	B = *(((uchar4 *)b) + i * (width >> 2) + j);
	//Bs[threadIdx.y * blockDim.x + threadIdx.x] = *(((uchar4 *)b) + i * (width >> 2) + j);

#if 0
	// Load 4 tiles

	if((j < (blockDim.x/4)) && (i < 16) && (j > (1920/4 - 4)) && (i > (800 - 16)))
		As[threadIdx.y * (blockDim.x + (4*2)) + threadIdx.x] = get_pixel_word(a,j*4-15,i-15,width,height);
	else
		As[threadIdx.y * (blockDim.x + (4*2)) + threadIdx.x] = *(((uchar4 *)a) + (i-15) * (width >> 2) + j*4-15);
	

	// Load remaining 2 tiles
	if(threadIdx.x < (2*4))
		As[threadIdx.y * (blockDim.x + (4*2)) + threadIdx.x + blockDim.x] = get_pixel_word(a, j*4 + ((blockDim.x/4) - 1)*16, i-15, width, height);

	__syncthreads();
#endif

	if((i < 800) && (j < (1920/4)))
	{

		if(threadIdx.y == 0 && threadIdx.x < (blockDim.x/4))
		{
			BEST_DIFFs[threadIdx.x] = 16*16*256;
		}
		__syncthreads();

		for (s=-15 ; s<16 ; s++)		// Search through a -15 to 15 neighborhood
			for (t=-15 ; t<16 ; t++)
			{
				int sad=calculate_sad(a,&B,0,0,j*4 + t,i + s,  width, height);	// Calculate difference between block from first image and second image
				SUMs[threadIdx.y * blockDim.x + threadIdx.x] = sad;
				__syncthreads();

				if((threadIdx.x & 3) == 0)	 //%4 // want 16*(blockDim.x/4) threads to run (16 in y direction), summation rowwise
				{
					for(k = 0; k < 4; k++)
					{
						partial_sum += SUMs[threadIdx.y * blockDim.x + threadIdx.x + k];
					}
					SUM_ROWs[threadIdx.y + blockDim.y*(threadIdx.x/(blockDim.x/4)) ] = partial_sum;
				}

				__syncthreads();
				if(threadIdx.y == 0 && threadIdx.x < (blockDim.x/4)) //sum up 16 partial sums using one thread in each tile
				{
					for(k = 0; k < 16; k++)
					{
						final_sum += SUM_ROWs[blockDim.y*threadIdx.x + k]; 
					}
					FINAL_SUMs[threadIdx.x] = final_sum; //store it in an array of number of tiles

					if (FINAL_SUMs[threadIdx.x] < BEST_DIFFs[threadIdx.x])			// If we found a better match then store it
					{
						BEST_Xs[threadIdx.x] = t;
						BEST_Ys[threadIdx.x] = s;
						BEST_DIFFs[threadIdx.x] = FINAL_SUMs[threadIdx.x];
					}

				}
				__syncthreads();

			}

		if((threadIdx.x < (blockDim.x/4)) && (threadIdx.y == 0))	
		{
			vx[blockIdx.y*width/16 + threadIdx.x + blockIdx.x*blockDim.x/4] = BEST_Xs[threadIdx.x];			// Store result
			vy[blockIdx.y*width/16 + threadIdx.x + blockIdx.x*blockDim.x/4] = BEST_Ys[threadIdx.x];
		}
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

__device__ uchar4 get_pixel_word(unsigned char* frame, int x, int y, unsigned int width, unsigned int height) 
{
	uchar4 buf, buf2;
	if((x >= 0) && (x < (width-3)) && (y >= 0) && (y < height))
	{
		buf = ((uchar4 *) frame)[(y * (width >> 2) + (x>>2))];
		if((x & 3))
		{
			buf2 = ((uchar4 *) frame)[(y * (width >> 2) + (x>>2)) + 1];

			switch(x&3)
			{
				case 1: buf.x = buf.y;
					buf.y = buf.z;
					buf.z = buf.w;
					buf.w = buf2.x;
					break;
				case 2: buf.x = buf.z;
					buf.y = buf.w;
					buf.z = buf2.x;
					buf.w = buf2.y;
					break;
				case 3: buf.x = buf.w;
					buf.y = buf2.x;
					buf.z = buf2.y;
					buf.w = buf2.z;
					break;
			}
		}
	}
	else
	{
		buf.x = get_pixel(frame,x,y,width,height);
		buf.y = get_pixel(frame,x+1,y,width,height);
		buf.z = get_pixel(frame,x+2,y,width,height);
		buf.w = get_pixel(frame,x+3,y,width,height);
	}
	return buf;
}

// This calculates the sum of absolute differences between two image blocks
// unsigned char*a 		Pointer to first image
// unsigned char*b 		Pointer to second image
// int ax,int ay			X and Y Position of the block in first image
// int bx,int by			X and Y Position of the block in second image
// int width			   Width of images
// int height			   Height of images
__device__ unsigned calculate_sad(unsigned char* a, uchar4* b,  int ax, int ay, int bx, int by, unsigned width, unsigned height)
{
	int sum=0;
	uchar4 ref_word;
	ref_word = get_pixel_word(a,bx,by,width,height);

//	uchar4 from_share = b[threadIdx.y * blockDim.x + threadIdx.x];
	uchar4 from_share = *b;
	sum += ABS( from_share.x - ref_word.x );
	sum += ABS( from_share.y - ref_word.y );
	sum += ABS( from_share.z - ref_word.z );
	sum += ABS( from_share.w - ref_word.w );

	return sum;



}

