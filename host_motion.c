#include "image.h"
#define ABS(x) ( (x) < 0 ? -(x) : (x) )

int calculate_sad(unsigned char* a,unsigned char *b,  unsigned int ax,unsigned int ay,unsigned int bx,unsigned int by,unsigned int stride)
{
	int sum=0;
	int i,j;

	for (i=0; i < 16; i++)
		for (j=0; j < 16; j++) 
			sum += ABS( b[ ax+j + (ay+i)*stride ] - a[ bx+j + (by+i)*stride ]);
	return sum;
}

void motion_search(unsigned char* a,unsigned char *b, unsigned width, unsigned height, int* vx,int* vy)
{
	int blocks_x=width/16;
	int blocks_y=height/16;
	int i,j,s,t;
	for (i=1; i < blocks_y-1 ; i++)
		for (j=1; j < blocks_x-1 ; j++)
		{
			int best_diff=100000;			
			int best_x,best_y=0;
			for (s=-15 ; s<16 ; s++)
				for (t=-15 ; t<16 ; t++)
				{
					int sad=calculate_sad(a,b,j*16,i*16,j*16+t,i*16+s,  width);
					if (sad < best_diff)
					{
						best_x = t;
						best_y = s;
						best_diff = sad;
					}
				}
			vx[j+i*blocks_x] = best_x;
			vy[j+i*blocks_x] = best_y;
		}
} 


int main()
{
	MyImage image_in_a;
	MyImage image_in_b;
	int i,j;

	int *image_mv_x;
	int *image_mv_y;

	readPgm((char *)"screenshot1.pgm", &image_in_a);

	readPgm((char *)"screenshot2.pgm", &image_in_b);

	printf("%d %d\n",image_in_b.width,image_in_b.height);
	

	image_mv_x = (int *) malloc(sizeof(int) * (image_in_a.width * image_in_a.height / 256)+1);
	image_mv_y = (int *) malloc(sizeof(int) * (image_in_b.width * image_in_b.height / 256)+1);

	if(!image_mv_x)
		return -1;

	printf("GSHJDSHVDH hyqfwyqfysfqy");

	motion_search(image_in_a.data,image_in_b.data,image_in_a.width, image_in_a.height,image_mv_x,image_mv_y);

	for (i = 0; i < image_in_a.width / 16; i++)
		for (j = 0; j < image_in_a.height / 16; j++)
			printf("%d,%d\n", image_mv_x[i+j*(image_in_a.height / 16)], image_mv_y[i+j*(image_in_a.height / 16)]);
	return 0;

}

