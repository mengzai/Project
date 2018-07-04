#pragma once


//-------------------------------------------------------------------------------------------------
// Logistic Regression: p(y=1|x) = 1/(1+exp(-w'*x-b))
// Let pi = 1/(1+exp(-yi*(w'*xi+b))), where yi = +1/-1
// LR    min                - 1/n*sum log(pi)
// L1-LR min     lam*1'*|w| - 1/n*sum log(pi)
// L2-LR min 0.5*lam*w'*w   - 1/n*sum log(pi)
class LR
{
public:
	LR(bool verbose = true);
	~LR();

	// w[num] or NULL, sample weights
	// X[num*dim]
	// y[num] +1/-1 or 1/0
	// reg 0: LR, 1: L1-LR, 2: L2-LR
	// lam lambda, regularization parameter for L1-LR and L2-LR
	bool TrainModel(int num, int dim, float *w, float *X, int *y, int reg, float lam);
	// non-negative weights LR
	bool TrainModelNNW(int num, int dim, float *w, float *X, int *y, int reg, float lam);

	// x[dim], f = w'*x+b
	// p(1|x) = 1/(1+exp(-w'*x-b))
	float TestModel(float *x, float &f);

	// X[num*dim]
	// y[num]
	// return accuracy rate
	float TestModel(int num, float *X, int *y, const char *filename);

	// get model [w b]
	float*GetModel(){ return wb; }
	void PrintModel();

	// filename, LR model file
	bool LoadModel(const char *filename);
	bool SaveModel(const char *filename);

	static double FuncGrad(void *cls, int n, double *x, double *g);

private:
	bool  verbose; // true default
	
	// training
	int   num; // training sample number
	int   dim; // feature dimension
	int   reg; // 0: LR, 1: L1-LR, 2: L2-LR
	float lam; // regularization parameter
	float *w;  // pointer to training sample weights
	float *X;  // pointer to training samples
	int   *y;  // pointer to labels

	// model
	float *wb; // model [w b]
};
