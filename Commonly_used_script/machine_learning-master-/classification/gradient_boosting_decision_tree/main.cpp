#include <stdio.h>
#include "data.h"
#include "gbrt.h"


//-------------------------------------------------------------------------------------------------
int main()
{
#if 0
	Data data;
	if(data.LoadDataTxt(false, "example_r_1.txt")) // false for regressor
	{
		GBRT gbrt;
		gbrt.SetParam(GBRT::R_SQUARE, 2, 1, 1);
		//gbrt.SetParam(GBRT::R_ABSOLUTE, 2, 1, 1);
		//gbrt.SetParam(GBRT::R_HUBER, 2, 1, 1, 0.5);
		gbrt.Train(data.n, data.d, data.X, data.yr, 100);
		
		float err = gbrt.Test(data.n, data.X, data.yr, NULL);
		printf("regressor error = %.3f\n", err);
	}
#endif

#if 1
	Data data;
	if(data.LoadDataTxt(true, "example_c.txt")) // true for classifier
	{
		//for(int i=0; i<data.n; i++) printf("y[%d] = %d\n", i, data.yc[i]);
		GBRT gbrt;
		if(1)
		{
			const int n = data.n, *yc = data.yc;
			float *y = new float[n];
			for(int i=0; i<n; i++) y[i] = (float)yc[i];
			
			//gbrt.SetParam(GBRT::C_LOGISTIC, 1, 1, 1);
			gbrt.SetParam(GBRT::C_RELU, 1, 1, 1);
			gbrt.Train(n, data.d, data.X, y, 100);
			
			gbrt.SaveModel("model.gbdt");
			delete[] y;
		}
		else gbrt.LoadModel("model.gbdt");
		
		float err = gbrt.Test(data.n, data.X, (float*)data.yc, NULL);
		printf("classifier error = %.3f\n", err);
	}
#endif

	return 0;
}
