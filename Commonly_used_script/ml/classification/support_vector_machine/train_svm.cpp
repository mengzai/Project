#include <time.h>
#include <stdio.h>
#include <string.h>
#include "data.h"
#include "svm.h"
#pragma warning(disable:4305)


//-------------------------------------------------------------------------------------------------
int main(int argn, const char *argv[])
{
	// read input parameters
	bool check = true;
	int datafile = 0, option = 0, verbose = 1;
	double c = 0;
	
	if(argn!=6){ if(argn>1) printf("Error: invalid number of parameters\n"); check = false; }
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
		if(sscanf(argv[2], "%d", &option)!=1 || !(0<=option && option<=3))
		{
			printf("option invalid\n");
			check = false;
		}
		
		// check c
		if(sscanf(argv[3], "%lf", &c)!=1 || c<0)
		{
			printf("c<0 invalid\n");
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
		printf("Usage: train_svm train_file option c model verbose\n");
		printf("       option  - 0:Hinge by FSMO; 1:Huber-Hinge by LBFGS; 2:Square-Hinge by LBFGS; 3:SVM+ by L-BFGS-B\n");
		printf("       verbose - 0:silent; 1:print progress messages\n");
		printf("       e.g. train_svm train.txt 0 1.0 model.svm 1\n");
		return 0;
	}

	// train SVM model
	SVM svm(verbose!=0);
	Data data;
	if(datafile==1 && data.LoadDataTxt(true, argv[1])
	|| datafile==2 && data.LoadDataBin(true, argv[1])
	|| datafile==3 && data.LoadDataSpa(true, argv[1]))
	{
		// (0,1) -> (-1,1)
		for(int i=0, n=data.n, *yc=data.yc; i<n; i++) if(yc[i]==0) yc[i] = -1;
	
		if(option<=2) svm.TrainLSVM(data.n, data.d, data.X, data.yc, c, option);
		else svm.TrainLSVMnnw(data.n, data.d, data.X, data.yc, c);
		if(verbose!=0) svm.PrintLSVM();

		svm.TestModel(data.n, data.X, data.yc, NULL);
		svm.SaveModel(argv[4]);
	}

	return 0;
}
