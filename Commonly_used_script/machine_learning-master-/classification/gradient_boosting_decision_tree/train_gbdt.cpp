#include <time.h>
#include <stdio.h>
#include <string.h>
#include "data.h"
#include "gbrt.h"
#pragma warning(disable:4305)


//-------------------------------------------------------------------------------------------------
int main(int argn, const char *argv[])
{
	// read input parameters
	bool check = true;
	int datafile = 0, option = 0, max_depth = 4, min_samples_leaf = 1, max_iters = 100;
	float learning_rate = 1.0f;
	
	if(argn!=8){ printf("Error: invalid number of parameters\n"); check = false; }
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
		if(sscanf(argv[2], "%d", &option)!=1 || !(3<=option && option<=4))
		{
			printf("option invalid\n");
			check = false;
		}
		
		// check max_depth
		if(sscanf(argv[3], "%d", &max_depth)!=1 || max_depth<=0)
		{
			printf("max_depth invalid\n");
			check = false;
		}

		// check min_samples_leaf
		if(sscanf(argv[4], "%d", &min_samples_leaf)!=1 || min_samples_leaf<=0)
		{
			printf("min_samples_leaf invalid\n");
			check = false;
		}
		
		// check max_iters
		if(sscanf(argv[5], "%d", &max_iters)!=1 || max_iters<=0)
		{
			printf("max_iters invalid\n");
			check = false;
		}		

		// check learning_rate
		if(sscanf(argv[6], "%f", &learning_rate)!=1 || !(0<learning_rate && learning_rate<=1))
		{
			printf("learning_rate invalid\n");
			check = false;
		}		
	}

	// print tip message
	if(!check)
	{
		printf("Usage: train_gbdt train_file option max_depth min_samples_leaf max_iters learning_rate model\n");
		printf("       option = 0:R_SQUARE, 1:R_ABSOLUTE, 2:R_HUBER, 3:C_LOGISTIC, 4:C_RELU; for classification, option>=3\n");
		printf("       e.g. train_gbdt example_c.txt 4 1 1 10 1.0f model.gbdt\n");
		return 0;
	}
	
	// train GBDT model
	GBRT gbrt;
	Data data;
	if(datafile==1 && data.LoadDataTxt(true, argv[1])
	|| datafile==2 && data.LoadDataBin(true, argv[1])
	|| datafile==3 && data.LoadDataSpa(true, argv[1]))
	{
		const int n = data.n, *yc = data.yc;
		float *y = new float[n];
		for(int i=0; i<n; i++) y[i] = yc[i]==1 ? 1 : -1;
		gbrt.SetParam((GBRT::LOSS)option, max_depth, min_samples_leaf, learning_rate);
		gbrt.Train(n, data.d, data.X, y, max_iters);
		delete[] y;

		//gbrt.Test(data.n, data.X, data.yc, NULL);
		gbrt.SaveModel(argv[7]);
	}

	return 0;
}
