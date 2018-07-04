#include <time.h>
#include <stdio.h>
#include <string.h>
#include "data.h"
#include "cart.h"
#pragma warning(disable:4305)


//-------------------------------------------------------------------------------------------------
int main(int argn, const char *argv[])
{
	// read input parameters
	bool check = true;
	int datafile = 0, option = 0, max_depth = 5, min_samples_leaf = 1;
	
	if(argn!=6){ printf("Error: invalid number of parameters\n"); check = false; }
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
		if(sscanf(argv[2], "%d", &option)!=1 || !(0<=option && option<=2))
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
	}

	// print tip message
	if(!check)
	{
		printf("Usage: train_cart train_file option max_depth min_samples_leaf model\n");
		printf("       option = 0:CART_C_GINI, 1:CART_C_CROSS_ENTROPY, 2:CART_C_MISCLASSIFICATION\n");
		printf("       e.g. train_cart train.txt 0 5 1 model.cart\n");
		return 0;
	}
	
	// train CART model
	CART cart;
	Data data;
	if(datafile==1 && data.LoadDataTxt(true, argv[1])
	|| datafile==2 && data.LoadDataBin(true, argv[1])
	|| datafile==3 && data.LoadDataSpa(true, argv[1]))
	{
		cart.SetParam((CART::CART_CRITERIA)option, max_depth, min_samples_leaf);
		cart.TrainClassifier(data.n, data.d, data.X, data.yc);

		cart.TestClassifier(data.n, data.X, data.yc);
		cart.SaveModel(argv[5]);
	}

	return 0;
}
