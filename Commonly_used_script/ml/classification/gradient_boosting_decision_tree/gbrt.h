#pragma once


//-------------------------------------------------------------------------------------------------
// Gradient Boosting Regression Tree (GBRT)
// F(x) = F0 + lr*F1(x) + ... + lr*Fm(x)
//-------------------------------------------------------------------------------------------------
class GBRT
{
public:
	GBRT();
	~GBRT();

	// R_... for regressor, C_... for classifier
	enum LOSS{ R_SQUARE, R_ABSOLUTE, R_HUBER, C_LOGISTIC, C_RELU };
	void SetParam(LOSS loss, int max_depth, int min_samples_leaf, float learning_rate, float
		 huber_quantile = 0.9f);
	
	void Train(int n, int d, float *X, float *y, int tmax);
	float Test(int n, float *X, float *y, float *f);
	
	bool LoadModel(const char *filename);
	bool SaveModel(const char *filename);

private:
	void Release();

	void TrainF0(int n, float *y, float *ym);
	void TrainFm(int m, int n, float *X, float *y, int *idn);
	void UpdateYm(int n, float *y, float *ym);
	
	// for regressor
	float GetMedian(int n, float *y);
	float GetQuantile(int n, float *y);
	float GetMedian(int n, float *y, bool *b, int m);
	float GetHuber(int n, float *y, bool *b, int m);
	
	// for classifier
	float GetLogistic(int n, float *y, bool *b, int m);
	float GetReLU(int n, float *y, bool *b, int m);

	int* SortFeatures(int n, float *X);
	
	struct treenode
	{
		int   dep;     // depth
		int   att;     // attribute index
		float thr;     // threshold
		float y;       // tree node response
		treenode *tn0; // left child
		treenode *tn1; // right child
	};
	bool SplitNode(int n, float *X, float *y, int *idn, bool *mask, treenode *tn);

	LOSS loss;
	int max_depth;
	int min_samples_leaf;
	float learning_rate;
	float huber_quantile;
	float*ytr;     // a pointer to training y
	
	int   d;       // feature dimension
	int   M;       // number of estimators
	float F0;      // initial estimator
	treenode **Fm; // M estimators
	float Qm;      // quantile
};
