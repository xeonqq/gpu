#define ABS(x) ( (x) < 0 ? -(x) : (x) )

__device__ unsigned char get_pixel(unsigned char* frame, int x, int y, unsigned width, unsigned height);
__device__ unsigned calculate_sad(unsigned char* a,unsigned char* b,  int ax, int ay, int bx, int by, unsigned width, unsigned height);

__global__ void  motion_search(unsigned char* a,unsigned char* b, unsigned int width, unsigned int height, int* vx, int* vy)
{
	int i = blockIdx.y*blockDim.y+threadIdx.y; // Block in Y axis
	int j = blockIdx.x*blockDim.x+threadIdx.x; // Block in X axis

	int blocks_x = width/16;
	int blocks_y = height/16;

	int s,t;
	int best_diff=16*16*256;			// This is larger than the largest possible absolute difference between two blocks
	int best_x,best_y=0;
	int k,l;


	unsigned char Bs[256];
	

	if((i < blocks_y) && (j < blocks_x) )
	{
		for (k=0; k<16; k++)
			for (l=0; l<16; l++)
				Bs[k*16 + l] = b[((i*16+k) * width) + (j*16+l)];

		for (s=-15 ; s<16 ; s++)		// Search through a -15 to 15 neighborhood
			for (t=-15 ; t<16 ; t++)
			{
				int sad=calculate_sad(a,Bs,0,0,j*16+t,i*16+s,  width, height);	// Calculate difference between block from first image and second image
				// Second image block shifted with (s,t)
				if (sad < best_diff)			// If we found a better match then store it
				{
					best_x = t;
					best_y = s;
					best_diff = sad;
				}
			}
		//		   printf("%i %i %f\n",best_x,best_y,best_diff/256.0f);  
		//printf("%i %i %f\n",best_x,best_y,best_diff/256.0f);  
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
__device__ unsigned calculate_sad(unsigned char* a, unsigned char* b,  int ax, int ay, int bx, int by, unsigned width, unsigned height)
{
	int sum=0;
	int i,j;
	uchar4 ref_word;
	for (i=0; i < 16; i++)
		for (j=0; j < 16; j+=4)
		{
			
			ref_word = get_pixel_word(a,bx+j,by+i,width,height);

			sum += ABS( b[i*16 + j] - ref_word.x );
			sum += ABS( b[i*16 + j+1] - ref_word.y );
			sum += ABS( b[i*16 + j+2] - ref_word.z );
			sum += ABS( b[i*16 + j+3] - ref_word.w );
		}
			//sum += ABS( b[i*16+j]/*get_pixel_current(b,j,i)*/ - get_pixel_ref(a,bx+j,by+i,width,height) );
	return sum;
}

