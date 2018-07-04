#include <time.h>
#include <stdio.h>
#include <string.h>
#include "data.h"
#include "sn.h"
#include "../logistic_regression/lr.h"
#pragma warning(disable:4305)


//-------------------------------------------------------------------------------------------------
float* load_sample_weights(const char *filename, int num, int *y)
{
	FILE *pfile = fopen(filename, "rt");
	if(pfile==NULL)
	{
		// is a float?
		float w0 = 0;
		if(sscanf(filename, "%f", &w0)==1)
		{
			float *w = new float[num];
			for(int i=0; i<num; i++) w[i] = y[i]==1 ? 1 : w0;
			return w;
		}
		return NULL;
	}

	// is a file!
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
	
	if(!(6<=argn && argn<=7)){ if(argn>1) printf("Error: invalid number of parameters\n"); check = false; }
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
		printf("Usage: train_sn trainfile [weights] option lambda model verbose\n");
		printf("       trainfile - training filename: *.txt, *.bin or *.spa\n");
		printf("       weights   - weights for training samples: float (weights for negative samples) or file (weights for all samples)\n");
		printf("       option    - 0:SN, 1:L1-SN, 2:L2-SN, 3:SN+, 4:L2-SN+\n");
		printf("       verbose   - 0:silent; 1:print progress messages\n");
		printf("       e.g. train_sn train.txt 2 0.001 model.sn 1\n");
		printf("       e.g. train_sn train.txt 3.0 2 0.001 model.sn 1\n");
		printf("       e.g. train_sn train.txt weights.txt 2 0.001 model.sn 1\n");
		return 0;
	}
	
	// train SN model
	SN sn(verbose!=0);
	Data data;
	if(datafile==1 && data.LoadDataTxt(true, argv[1])
	|| datafile==2 && data.LoadDataBin(true, argv[1])
	|| datafile==3 && data.LoadDataSpa(true, argv[1]))
	{
		// load sample weights
		float *w = NULL;
		if(argn==7)
		{
			w = load_sample_weights(argv[2], data.n, data.yc);
			if(w==NULL)
			{
				printf("load_sample_weights(\"%s\", %d) failed\n", argv[2], data.n);
				return 3;
			}
		}

		LR lr(verbose!=0);
		if(option<=2) lr.TrainModel(data.n, data.d, w, data.X, data.yc, option, lambda);
		else lr.TrainModelNNW(data.n, data.d, w, data.X, data.yc, option==3?0:2, lambda);

		//if(option<=2) sn.TrainModel(data.n, data.d, w, data.X, data.yc, option, lambda);
		//else sn.TrainModelNNW(data.n, data.d, w, data.X, data.yc, option==3?0:2, lambda);
		if(option<=2) sn.TrainModel(data.n, data.d, w, data.X, data.yc, option, lambda, lr.GetModel());
		else sn.TrainModelNNW(data.n, data.d, w, data.X, data.yc, option==3?0:2, lambda, lr.GetModel());
		if(verbose!=0) sn.PrintModel();

		sn.TestModel(data.n, data.X, data.yc, NULL);
		sn.SaveModel(argv[argn-2]);

		// release sample weights
		if(w!=NULL){ delete[] w; w = NULL; }
	}

	return 0;
}
