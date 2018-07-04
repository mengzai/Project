#include <time.h>
#include <stdio.h>
#include <string.h>
#include "data.h"
#include "nn.h"


//-------------------------------------------------------------------------------------------------
int main(int argn, const char *argv[])
{
	// read input parameters
	bool check = true;
	int datafile = 0;
	
	if(!(3<=argn && argn<=4)){ if(argn>1) printf("Error: invalid number of parameters\n"); check = false; }
	else
	{
		// check suffix of test_file
		const char *suffix = argv[1]+strlen(argv[1])-3;
		if(strcmp(suffix, "txt")==0) datafile = 1;
		else if(strcmp(suffix, "bin")==0) datafile = 2;
		else if(strcmp(suffix, "spa")==0) datafile = 3;
		else
		{
			printf("test_file %s must be *.txt, *.bin or *.spa\n", argv[1]);
			check = false;
		}
	}

	// print tip message
	if(!check)
	{
		printf("Usage: test_nn testfile model [result]\n");
		printf("       e.g. test_nn test_mnist.bin mnist.nn\n");
		printf("       e.g. test_nn test_mnist.bin mnist.nn result.txt\n");
		return 0;
	}

	// load NN model
	NN nn;
	if(!nn.LoadModel(argv[2]))
	{
		printf("Error: failed to load model %s\n", argv[2]);
		return 0;
	}
	
	// test NN model
	Data data;
	if(datafile==1 && data.LoadDataTxt(true, argv[1])
	|| datafile==2 && data.LoadDataBin(true, argv[1])
	|| datafile==3 && data.LoadDataSpa(true, argv[1]))
	{
		// (-1,1) -> (0,1)
		for(int i=0, n=data.n, *yc=data.yc; i<n; i++) if(yc[i]==-1) yc[i] = 0;

		if(argn==3) nn.TestModel(data.n, data.X, data.yc);
		else
		{
			FILE *pfile = fopen(argv[3], "wt");
			if(pfile==NULL)
			{
				printf("Error: failed to open %s for writing\n", argv[3]);
				return 0;
			}
			const int n = data.n, d = data.d, c = nn.GetOutputNumber();
			float *f = nn.GetOutputs();
			int *yc = data.yc, cor = 0;
			for(int i=0; i<n; i++)
			{
				int y = nn.TestModel(data.X+i*d);
				if(yc[i]==y) cor++;
				for(int k=0; k<c; k++) fprintf(pfile, k+1<c ? "%f\t" : "%f\n", f[k]);
			}
			float acc = (float)cor/(float)n;
			printf("* accuracy  = %.2f%% (%d/%d)\n", 100*acc, cor, n);
			fclose(pfile);
		}
	}

	return 0;
}
