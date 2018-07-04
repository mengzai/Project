#include <stdio.h>
#include "lr.h"


//-------------------------------------------------------------------------------------------------
extern "C" int TrainLR(int n, int d, float *w, float *X, int *y, int opti, float lamb, int verb, 
char *filename)
{
	LR lr(verb!=0);
	
	if(opti<=2) lr.TrainModel(n, d, w, X, y, opti, lamb);
	else lr.TrainModelNNW(n, d, w, X, y, opti==3?0:2, lamb);
	lr.PrintModel();

	lr.TestModel(n, X, y, NULL);
	lr.SaveModel(filename);
	
	return 0;
}


//-------------------------------------------------------------------------------------------------
extern "C" int TestLR(char *filename, int n, int d, float *X, float *f)
{
	LR lr(false);
	if(!lr.LoadModel(filename)) return 1;
	
	for(int i=0; i<n; i++) lr.TestModel(X+i*d, f[i]);	
	
	return 0;
}