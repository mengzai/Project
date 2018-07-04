#pragma once
#include <stdio.h>


//-------------------------------------------------------------------------------------------------
// Classification and Regression Trees (CART)
//-------------------------------------------------------------------------------------------------
class CART
{
public:
	CART();
	~CART();

	enum CART_CRITERIA
	{
		CART_C_GINI               = 0,
		CART_C_CROSS_ENTROPY      = 1,
		CART_C_MISCLASSIFICATION  = 2,
		CART_R_MEAN_SQUARED_ERROR = 3
	};
	void SetParam(CART_CRITERIA criteria, int max_depth, int min_samples_leaf);
	// X[d*n] features, y[n] labels, z[n] sample weights
	void TrainClassifier(int n, int d, float *X, int *y, float *z = NULL);
	int  TestClassifier(int n, float *X, int *y, float *z = NULL); // return error number

	void TrainRegressor(int n, int d, float *X, float *y, float *z = NULL);
	float TestRegressor(int n, float *X, float *y, float *z = NULL); // return mean_abs_err

	bool SaveModel(const char *filename);
	bool LoadModel(const char *filename);

private:
	void Release();
	void SortFeatures();
	bool SplitNode(bool *mask, int &attr, float &thre, float &impu, float *other);

	inline float GetNodeImpurity(int m, int *p);
	inline float GetNodeImpurity(int m, int *p0, int *p1);

	void DrawCART(); // draw cart tree

	CART_CRITERIA criteria;
	int max_depth;
	int min_samples_leaf;

	int    d;   // dimension
	int    c;   // class number for classifier

	// for training stage
	int    n;   // sample number
	int   *yc;  // yc[n] label vector for classifier, 0,...,c-1
	float *yr;  // yr[n] target vector for regressor
	float *X;   // X[n*d] feature matrix
	float *z;   // z[n]  sample weights
	int   *idn; // i[d*n] sort features by ascending order

	// cart tree model
	struct treenode
	{
		int     size; // samples
		int     dept; // depth
		int     attr; // attribute
		float   thre; // threshold
		float   impu; // impurity
		int       yc; // classifier
		float     yr; // regressor
		treenode*tn0; // left  child
		treenode*tn1; // right child
		bool   *mask; // subset mask
	}*cart;           // cart model
};
