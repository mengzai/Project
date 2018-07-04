#pragma once


//-------------------------------------------------------------------------------------------------
// Single Neuron: p(y=1|x) = 1/(1+exp(-w'*x-b))
// Let pi = 1/(1+exp(yi*(w'*xi+b))), where yi = +1/-1
// SN    min                + 1/n*sum pi
// L1-SN min     lam*1'*|w| + 1/n*sum pi
// L2-SN min 0.5*lam*w'*w   + 1/n*sum pi
class SN
{
public:
	SN(bool verbose = true);
	~SN();

	// w[num] or NULL, sample weights
	// X[num*dim]
	// y[num] +1/-1 or 1/0
	// reg 0: SN, 1: L1-SN, 2: L2-SN
	// lam lambda, regularization parameter for L1-SN and L2-SN
	bool TrainModel(int num, int dim, float *w, float *X, int *y, int reg, float lam, float *wb = 0);
	bool TrainModelNNW(int num, int dim, float *w, float *X, int *y, int reg, float lam, float *wb = 0);

	// x[dim]
	// p(1|x) = 1/(1+exp(-w'*x-b))
	float TestModel(float *x);

	// X[num*dim]
	// y[num]
	// return accuracy rate
	float TestModel(int num, float *X, int *y, const char *filename);

	// get model [w b]
	float*GetModel(){ return wb; }
	void PrintModel();

	// filename, model file
	bool LoadModel(const char *filename);
	bool SaveModel(const char *filename);

	static double FuncGrad(void *cls, int n, double *x, double *g);

private:
	bool  verbose; // true default
	
	// training
	int   num; // training sample number
	int   dim; // feature dimension
	int   reg; // 0: SN, 1: L1-SN, 2: L2-SN
	float lam; // regularization parameter
	float *w;  // pointer to training sample weights
	float *X;  // pointer to training samples
	int   *y;  // pointer to labels

	// model
	float *wb; // model [w b]
};
