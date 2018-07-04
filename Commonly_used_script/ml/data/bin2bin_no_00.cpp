#include <stdio.h>
#include "data.h"


//-------------------------------------------------------------------------------------------------
bool LoadLabelPredict(const char *filename, int num, bool *keep)
{	
	FILE *pfile = fopen(filename, "rt");
	if(pfile==NULL) return false;
	
	bool ret = true;
	for(int i=0; i<num; i++)
	{
		int   y = 0;
		float p = 0;
		if(fscanf(pfile, "%d%f", &y, &p)!=2)
		{
			printf("%s: row %d is invalid\n", filename, i);
			ret = false;
			break;
		}
		
		if(y!=1 && p<0.5f) keep[i] = false;
		else keep[i] = true;
	}
	
	fclose(pfile);	
	return ret;
}


//-------------------------------------------------------------------------------------------------
int main(int argn, const char *argv[])
{
	if(argn<4)
	{
		printf("Usage: bin2bin_no_00 file.bin label_predict.txt new_file.bin\n");
		return 0;
	}
	
	Data data;
	if(data.LoadDataBin(true, argv[1]))
	{
		bool *keep = new bool[data.n];
		if(LoadLabelPredict(argv[2], data.n, keep))
		{
			FILE *pfile = fopen(argv[3], "wb");
			if(pfile==NULL) printf("fopen(%s) failure\n", argv[3]);
			else
			{
				int n = data.n, d = data.d, n1 = 0;
				fwrite(&n, sizeof(int), 1, pfile);
				fwrite(&d, sizeof(int), 1, pfile);
				for(int i=0; i<n; i++)
				{
					if(!keep[i]) continue;
					fwrite(data.yc+i, sizeof(int), 1, pfile);
					fwrite(data.X+i*d, sizeof(float), d, pfile);
					n1++;
				}
				fseek(pfile, 0, SEEK_SET);
				fwrite(&n1, sizeof(int), 1, pfile);

				fclose(pfile);
				printf("finished conversion: %s => %s\n", argv[1], argv[3]);
			}
		}

		if(keep!=NULL) delete[] keep;
	}

	return 0;
}