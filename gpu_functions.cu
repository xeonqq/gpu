#define ABS(x) ( (x) < 0 ? -(x) : (x) )

__device__ unsigned char get_pixel(unsigned char* frame, int x, int y, unsigned width, unsigned height);
__device__ unsigned calculate_sad(unsigned int* a,unsigned char* b,  int ax, int ay, int bx, int by, unsigned width, unsigned height);

__shared__ unsigned int count_ref;
__shared__ unsigned int word_ref;
__shared__ unsigned int count;
__shared__ unsigned int word;

__global__ void  motion_search(unsigned int* a,unsigned int* b, unsigned int width, unsigned int height, int* vx, int* vy)
{
	//int j = blockIdx.y*blockDim.y+threadIdx.y; // Block in X axis
	//int i = blockIdx.x*blockDim.x+threadIdx.x; // Block in Y axis
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
		//for (k=0; k<16; k++)
		//	for (l=0; l<16; l++)
		//		Bs[k*16 + l] = b[((i*16+k) * width) + (j*16+l)];
		for (k=0; k<16; k++)
			for (l=0; l<4; l++)
			{
				int reg = b[((i*16+k) * (width/4)) + (j*4+l)];
				//Bs[blockDim.x*4*(threadIdx.y*16 + k) + l + (threadIdx.x*4)] = b[((i*16+k) * (width/4)) + (j*4+l)];
				Bs[k*16+4*l] = reg & 0xFF;
				Bs[k*16+4*l+1] = (reg >> 8) & 0xFF;
				Bs[k*16+4*l+2] = (reg >> 16) & 0xFF;
				Bs[k*16+4*l+3] = (reg >> 24) & 0xFF;
			}

		count_ref = 0;
		count = 0;

		for (s=-15 ; s<16 ; s++)		// Search through a -15 to 15 neighborhood
			for (t=-15 ; t<16 ; t++)
			{
				int sad=calculate_sad(a,Bs,0,0,j*16+t,i*16+s,  width, height);	// Calculate difference between block from first image and second image
				//int sad=calculate_sad(a,b,j*16,i*16,j*16+t,i*16+s,  width, height);	// Calculate difference between block from first image and second image
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
__device__ unsigned char get_pixel_ref(unsigned int* frame, int x, int y, unsigned width, unsigned height)
{
	unsigned char boundary = count_ref & 3;

#if 0
	unsigned char shift = 0;
	bool edge = false;
	if (x >= width)
	{ 
		x = width-1;
		shift = 24;
		edge = true;
	}
	if (x < 0) 
	{
		x = 0;
		shift = 0;
		edge = true;
	}
	if (y >= height)
	{
		y = height-1;
		shift = (x & 3) << 3;
		edge = true;
	}
	if (y < 0)
	{ 
		y = 0; 
		shift = (x & 3) << 3;
		edge = true;
	}

	if(edge)
		return((frame[(y * (width>>2)) + (x>>2)] >> shift) & 0xFF);
#else
	if (x >= width)
		if(y>0 && y<height)
			return((frame[(y*(width>>2)) + ((width>>2) - 1)] >> 24) & 0xFF);
		else if(y<0)	
			return((frame[(width>>2) - 1]  >> 24) & 0xFF);
		else return((frame[(height-1)*(width>>2) + ((width>>2) - 1)] >> 24) & 0xFF);
		
	if (x < 0)
		if(y>0 && y<height)
			return((frame[y*(width>>2)]) & 0xFF);
		else if(y<0)	
			return(frame[0] & 0xFF);
		else return((frame[(height-1)*(width>>2)]) & 0xFF);
			
	if (y >= height)
		if(x>0 && x<width)
			return((frame[(height-1)*(width>>2) + (x>>2)] >> ((x&3)<<3)) & 0xFF);
		else if(x<0)
			return((frame[(height-1)*(width>>2)]) & 0xFF);
		else return((frame[(height-1)*(width>>2) + ((width>>2) - 1)] >> 24) & 0xFF);
			
	if (y < 0)
		if(x>0 && x<width)
			return((frame[x>>2] >> ((x&3)<<3)) & 0xFF);
		else if(x<0)
			return((frame[0]) & 0xFF);
		else return((frame[(width>>2) - 1] >> 24) & 0xFF);
#endif

	if(!boundary)
		word_ref = frame[(y * (width>>2)) + (x>>2)];
	else
		count_ref++;

	return((word_ref >> ((x&3)<<3)) & 0xFF);
}

__device__ unsigned char get_pixel_current(unsigned int* frame, int x, int y)
{
	unsigned char boundary = count & 3;

	if(!boundary)
		word = frame[y*4 + (x>>2)];
	else
		count++;

	return((word >> (boundary*8)) & 0xFF);
}

// This calculates the sum of absolute differences between two image blocks
// unsigned char*a 		Pointer to first image
// unsigned char*b 		Pointer to second image
// int ax,int ay			X and Y Position of the block in first image
// int bx,int by			X and Y Position of the block in second image
// int width			   Width of images
// int height			   Height of images
__device__ unsigned calculate_sad(unsigned int* a,unsigned char* b,  int ax, int ay, int bx, int by, unsigned width, unsigned height)
{
	int sum=0;
	int i,j;
	count = 0;
	for (i=0; i < 16; i++)
		for (j=0; j < 16; j++)  
			//sum += ABS( b[i*16+j] - get_pixel(a,bx+j,by+i,width,height) );
			sum += ABS( b[i*16+j]/*get_pixel_current(b,j,i)*/ - get_pixel_ref(a,bx+j,by+i,width,height) );
	return sum;
}

