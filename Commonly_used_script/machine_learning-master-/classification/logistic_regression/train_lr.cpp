#include <time.h>
#include <stdio.h>
#include <string.h>
#include "data.h"
#include "lr.h"
#pragma warning(disable:4305)


//-------------------------------------------------------------------------------------------------
float* load_sample_weights(const char *filename, int num)
{
	FILE *pfile = fopen(filename, "rt");
	if(pfile==NULL) return NULL;
	
	float *w = new float[num];
	for(int i=0; i<num; i++)
	{
		if(fscanf(pfile, "%f\n", w+i)!=1)
		{
			delete[] w;
			w = NULL;
			break;
		}
	}
	
	fclose(pfile);
	return w;
}


//-------------------------------------------------------------------------------------------------
int main(int argn, const char *argv[])
{
	// read input parameters
	bool check = true;
	int datafile = 0, option = 0, verbose = 1;
	double lambda = 0;
	
	if(!(6<=argn && argn<=7)){ printf("Error: invalid number of parameters\n"); check = false; }
	else
	{
		// check suffix of train_file
		const char *suffix = argv[1]+strlen(argv[1])-3;
		if(strcmp(suffix, "txt")==0) datafile = 1;
		else if(strcmp(suffix, "bin")==0) datafile = 2;
		else if(strcmp(suffix, "spa")==0) datafile = 3;
		else
		{
			printf("train_file %s must be *.txt, *.bin or *.spa\n", argv[1]);
			check = false;
		}
		
		// check option
		if(sscanf(argv[argn-4], "%d", &option)!=1 || !(0<=option && option<=4))
		{
			printf("option invalid\n");
			check = false;
		}
		
		// check lambda
		if(sscanf(argv[argn-3], "%lf", &lambda)!=1 || lambda<0)
		{
			printf("lambda<0 invalid\n");
			check = false;
		}

		// check verbose
		if(sscanf(argv[argn-1], "%d", &verbose)!=1 || !(0<=verbose && verbose<=1))
		{
			printf("verbose invalid\n");
			check = false;
		}
	}

	// print tip message
	if(!check)
	{
		printf("Usage: train_lr train_file [weights_file] option(0:LR, 1:L1-LR, 2:L2-LR, 3:LR+, 4:L2-LR+) lambda model verbose(0,1)\n");
		printf("       e.g. train_lr train.txt 2 0.001 model.lr 1\n");
		printf("       e.g. train_lr train.txt weights.txt 2 0.001 model.lr 1\n");
		return 1;
	}
	
	// train LR model
	LR lr(verbose!=0);
	Data data;
	if(datafile==1 && data.LoadDataTxt(true, argv[1])
	|| datafile==2 && data.LoadDataBin(true, argv[1])
	|| datafile==3 && data.LoadDataSpa(true, argv[1]))
	{
		// load sample weights
		float *w = NULL;
		if(argn==7)
		{
			w = load_sample_weights(argv[2], data.n);
			if(w==NULL)
			{
				printf("load_sample_weights(\"%s\", %d) failed\n", argv[2], data.n);
				return 3;
			}
		}

		if(option<=2) lr.TrainModel(data.n, data.d, w, data.X, data.yc, option, lambda);
		else lr.TrainModelNNW(data.n, data.d, w, data.X, data.yc, option==3?0:2, lambda);
		lr.PrintModel();

		lr.TestModel(data.n, data.X, data.yc, NULL);
		lr.SaveModel(argv[argn-2]);
		
		// release sample weights
		if(w!=NULL){ delete[] w; w = NULL; }
		return 0;
	}
	else return 2;
}
