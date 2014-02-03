/*

	motion.c

	Short: Full Search for full-pel motion vectors in the luma image. After that the image is reconstructed from the reference image.
	Long: This program takes two images "frameA.yuv" and "frameB.yuv" and searches for motion vector for each 16x16 block of the image.
			For each block of "frameB.yuv" it search for the block in "frameA.yuv" that is the closest match in a -15 to 15 neighborhood of the block. 
			After that is tries to reconstruct "frameB.yuv" from the motion vectors and "frameA.yuv" and writes this output to "output.yuv"
		   

Display individual frames using:

	display -colorspace RGB -size 1920x800 -depth 8 FILENAME

Show all frames:

	display -loop 0 -delay 1 -colorspace RGB -size 1920x800 -depth 8 frameA.yuv output.yuv frameB.yuv
*/

#include <stdio.h>
#include <stdlib.h>

#define ABS(x) ( (x) < 0 ? -(x) : (x) )


// Gets a pixel from the Luma plane of the image, takes care of boundaries
unsigned char get_pixel(unsigned char* frame, int x, int y, unsigned width, unsigned height) 
{
	if (x >= width) x=width-1;
	if (x < 0) x=0;
	if (y >= height) y=height-1;
	if (y < 0) y=0;
	return frame[x+y*width];
}


// Gets a pixel from one of the chroma planes of the image, takes care of boundaries, interpolates when needed
// This is only used for reconstruction, you do not need to care about bringing this function to the GPU
unsigned char get_pixel_chroma(unsigned char* frame, int x, int y, unsigned width, unsigned height, unsigned int plane) 
{
	unsigned char* cframe=&(frame[width*height+(width*height/4)*plane]);
	if (x >= width) x=width-1;
	if (x < 0) x=0;
	if (y >= height) y=height-1;
	if (y < 0) y=0;
	return cframe[(x/2)+(y/2)*(width/2)];	
	if (x % 2) {
		if (y %2) {
			return (cframe[(x/2)+(y/2)*(width/2)]+cframe[(x/2)+(y/2+1)*(width/2)]+cframe[(x/2+1)+(y/2)*(width/2)]+cframe[(x/2+1)+(y/2+1)*(width/2)])/4;	
		} else {
			return (cframe[(x/2)+(y/2)*(width/2)]+cframe[(x/2+1)+(y/2)*(width/2)])/2;	
		}
	} else {
		if (y %2) {
			return (cframe[(x/2)+(y/2)*(width/2)]+cframe[(x/2)+(y/2+1)*(width/2)])/2;	
		} else {
			return cframe[(x/2)+(y/2)*(width/2)];	
		}
	}	
}


// This calculates the sum of absolute differences between two image blocks
// unsigned char*a 		Pointer to first image
// unsigned char*b 		Pointer to second image
// int ax,int ay			X and Y Position of the block in first image
// int bx,int by			X and Y Position of the block in second image
// int width			   Width of images
// int height			   Height of images
unsigned calculate_sad(unsigned char* a,unsigned char* b,  int ax, int ay, int bx, int by, unsigned width, unsigned height)
{
	int sum=0;
	for (int i=0; i < 16; i++)
		for (int j=0; j < 16; j++)  
			sum += ABS( get_pixel(b,ax+j,ay+i,width,height) - get_pixel(a,bx+j,by+i,width,height) );
	return sum;
}



// This is the main motion search function that needs to be moved to the GPU
// unsigned char*a 		Pointer to first image
// unsigned char*b 		Pointer to second image
// int width			   Width of images
// int height			   Height of images
// int* vx					Array for X Part of the motion vectors
// int* vy					Array for Y Part of the motion vectors
void motion_search(unsigned char* a,unsigned char* b, unsigned width, unsigned height, int* vx, int* vy)
{
	int blocks_x=width/16;
	int blocks_y=height/16;
	for (int i=0; i < blocks_y ; i++)		// For all blocks in Y direction
	{
	 //  printf(".");
		for (int j=0; j < blocks_x ; j++)   // For all blocks in X direction
		{
			int best_diff=16*16*256;			// This is larger than the largest possible absolute difference between two blocks
			int best_x,best_y=0;
			for (int s=-15 ; s<16 ; s++)		// Search through a -15 to 15 neighborhood
				for (int t=-15 ; t<16 ; t++)
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
} 


// Loads a pictures
void load_picture(unsigned char** dest, char* filename)
{
	FILE *f=fopen(filename,"rb");
	if (f == NULL) {
		printf("\nERROR: Could not open %s ...",filename);
		exit(-1);
	}
	*dest = (unsigned char*)malloc(1920*800*3/2);
	fread(*dest,1920*800*3/2,1,f);
	fclose(f);
}


// Reconstructions a image from reference image and motion vectors
// This is just to show the concept, you do not need to move this to the GPU or change anything about this
void reconstruct_image(unsigned char* a, unsigned width, unsigned height, int* vx, int* vy, char* filename) 
{
	int blocks_x=width/16;
	unsigned char* output=(unsigned char*)malloc(width*height*3/2);
	for (int i=0; i< height; i++)
   {
		for (int j=0; j< width; j++)
		{
			unsigned mb=(j/16)+(i/16)*blocks_x;
			int mx=vx[mb];
			int my=vy[mb];
			output[j+i*width]=get_pixel(a,j+mx,i+my,width,height);
		}
	}
	for (int i=0; i< height/2; i++)
   {
		for (int j=0; j< width/2; j++)
		{
			unsigned mb=(j/8)+(i/8)*blocks_x;
			int mx=vx[mb];
			int my=vy[mb];
			output[j+i*(width/2)+width*height]=get_pixel_chroma(a,2*j+mx,2*i+my,width,height,0);
			output[j+i*(width/2)+width*height+(width*height/4)]=get_pixel_chroma(a,2*j+mx,2*i+my,width,height,1);
		}
	}
	FILE* f=fopen(filename,"wb");
	fwrite(output,1920*800*3/2,1,f);
	free(output);
}

int main(int argc,char **args)
{
	unsigned char* ref_frame;
	unsigned char* current_frame;
	int num_blocks=(1920*800)/(16*16);
	int vx[num_blocks];				// Reserve some memory for motion vectors
	int vy[num_blocks];
	load_picture(&ref_frame,"frameA.yuv");							// Load pictures
	load_picture(&current_frame,"frameB.yuv");
   motion_search(ref_frame,current_frame,1920,800,vx,vy);	// Search for motion vectors
	for(int i=0;i<num_blocks;i++) 									// Display motion vectors
	{	
		printf("\n X:%i Y:%i",vx[i],vy[i]);
	}
	reconstruct_image(ref_frame,1920,800,vx,vy,"output.yuv");	// Reconstruct image
	free(current_frame);
	free(ref_frame);
}
