#include <stdio.h>

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

