#include <stdio.h>
#include "data.h"


//-------------------------------------------------------------------------------------------------
int main(int argn, const char *argv[])
{
	if(argn<3)
	{
		printf("Usage: txt2bin file.txt file.bin\n");
		return 0;
	}
	
	Data data;
	if(data.LoadDataTxt(true, argv[1]))
	{
		FILE *pfile = fopen(argv[2], "wb");
		if(pfile==NULL)
		{
			printf("fopen(%s) failure\n", argv[2]);
			return 0;
		}
		
		const int n = data.n, d = data.d;
		fwrite(&n, sizeof(int), 1, pfile);
		fwrite(&d, sizeof(int), 1, pfile);
		for(int i=0; i<n; i++)
		{
			fwrite(data.yc+i, sizeof(int), 1, pfile);
			fwrite(data.X+i*d, sizeof(float), d, pfile);
		}
		
		fclose(pfile);
		printf("finished conversion: %s => %s\n", argv[1], argv[2]);
	}
		
	return 0;
}