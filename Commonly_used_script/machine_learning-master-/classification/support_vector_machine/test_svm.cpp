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
	int datafile = 0;
	
	if(argn!=3){ printf("Error: invalid number of parameters\n"); check = false; }
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
		printf("Usage: test_svm test_file model\n");
		printf("       e.g. test_svm test.txt model.svm\n");
		return 0;
	}

	// load SVM model
	SVM svm;
	if(!svm.LoadModel(argv[2]))
	{
		printf("Error: failed to load model %s\n", argv[2]);
		return 0;
	}
	
	// train SVM model
	Data data;
	if(datafile==1 && data.LoadDataTxt(true, argv[1])
	|| datafile==2 && data.LoadDataBin(true, argv[1])
	|| datafile==3 && data.LoadDataSpa(true, argv[1]))
	{
		svm.TestModel(data.n, data.X, data.yc);
	}

	return 0;
}
