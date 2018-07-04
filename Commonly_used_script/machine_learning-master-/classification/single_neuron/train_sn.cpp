#include <time.h>
#include <stdio.h>
#include <string.h>
#include "data.h"
#include "sn.h"
#pragma warning(disable:4305)


//-------------------------------------------------------------------------------------------------
int main(int argn, const char *argv[])
{
	// read input parameters
	bool check = true;
	int datafile = 0, option = 0;
	double lambda = 0;
	
	if(argn!=5){ printf("Error: invalid number of parameters\n"); check = false; }
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
		if(sscanf(argv[2], "%d", &option)!=1 || !(0<=option && option<=4))
		{
			printf("option invalid\n");
			check = false;
		}
		
		// check lambda
		if(sscanf(argv[3], "%lf", &lambda)!=1 || lambda<0)
		{
			printf("lambda<0 invalid\n");
			check = false;
		}
	}

	// print tip message
	if(!check)
	{
		printf("Usage: train_sn train_file option(0:SN, 1:L1-SN, 2:L2-SN, 3:SN+, 4:L2-SN+) lambda model\n");
		printf("       e.g. train_sn train.txt 2 0.001 model.sn\n");
		return 0;
	}
	
	// train SN model
	SN sn;
	Data data;
	if(datafile==1 && data.LoadDataTxt(true, argv[1])
	|| datafile==2 && data.LoadDataBin(true, argv[1])
	|| datafile==3 && data.LoadDataSpa(true, argv[1]))
	{
		if(option<=2) sn.TrainModel(data.n, data.d, data.X, data.yc, option, lambda);
		else sn.TrainModelNNW(data.n, data.d, data.X, data.yc, option==3?0:2, lambda);
		sn.PrintModel();		

		sn.TestModel(data.n, data.X, data.yc);
		sn.SaveModel(argv[4]);
	}

	return 0;
}
