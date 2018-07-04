#include <stdio.h>
#include <string.h>
#include "data.h"
#include "nn.h"


//-------------------------------------------------------------------------------------------------
int main(int argn, const char *argv[])
{
	// read input parameters
	bool check = true;
	int datafile = 0, hidden = 0, layers[8] = {0}, option = 0, epochs = 0;
	double lambda = 0;
	
	if(argn!=7){ if(argn>1) printf("Error: invalid number of parameters\n"); check = false; }
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
		
		// check hidden
		int len = (int)strlen(argv[2]);
		const char *buf = argv[2];
		for(int i=0; i<len; i++)
		{
			int j = i;
			while(j<len && '0'<=buf[j] && buf[j]<='9') j++;
			if(j>i && (j==len || j+1<len) && hidden<6)
			{
				sscanf(buf+i, "%d", layers+hidden+1);
				hidden++;
			}
			else
			{
				printf("hidden invalid\n");
				check = false;
				break;
			}
			i = j;
		}

		// check option
		if(sscanf(argv[3], "%d", &option)!=1 || !(0<=option && option<=4))
		{
			printf("option invalid\n");
			check = false;
		}
		
		// check lambda
		if(sscanf(argv[4], "%lf", &lambda)!=1 || lambda<0)
		{
			printf("lambda invalid\n");
			check = false;
		}
		
		// check epochs
		if(sscanf(argv[5], "%d", &epochs)!=1 || epochs<=0)
		{
			printf("epochs invalid\n");
			check = false;
		}
	}

	// print tip message
	if(!check)
	{
		printf("Usage: train_nn trainfile hidden option lambda epochs model\n");
		printf("       trainfile - training filename: *.txt, *.bin or *.spa\n");
		printf("       hidden    - units in hidden layers for NN, e.g. 0:two layers, 100:three layers, 100+100:four layers\n");
		printf("       option    - 0:Sigmoid=>Squared; 1:Sigmoid=>Softmax; 2:ReLU=>Hinge; 3:ReLU=>SquaredHinge; 4:ReLU=>SmoothHinge\n");
		printf("       lambda    - regularization term 0.5*λ*w'*w\n");
		printf("       epochs    - maximum training epochs\n");
		printf("       model     - saved model filename\n");
		printf("       e.g. train_nn train_mnist.bin 100 3 0.001 50 mnist.nn\n");
		return 0;
	}

	// train NN model
	Data data;
	if(datafile==1 && data.LoadDataTxt(true, argv[1])
	|| datafile==2 && data.LoadDataBin(true, argv[1])
	|| datafile==3 && data.LoadDataSpa(true, argv[1]))
	{
		// (-1,1) -> (0,1)
		int c = 0;
		for(int i=0, n=data.n, *yc=data.yc; i<n; i++)
		{
			if(yc[i]==-1) yc[i] = 0;
			if(c<yc[i]) c = yc[i];
		}
		
		if(hidden==1 && layers[1]==0) hidden = 0; // two layers

		NN nn;
		nn.opt = (NN::ACTIVATION_LOSS)option;
		nn.lam = lambda;
		layers[0] = data.d; // input layer
		layers[1+hidden] = c==1 ? 1 : c+1; // output layer
		nn.Setup(1+hidden+1, layers); // all layers
		nn.TrainModelBySGD(data.n, data.X, data.yc, epochs);

		nn.SaveModel(argv[6]);
	}

	return 0;
}
