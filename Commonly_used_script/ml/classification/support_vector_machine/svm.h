#pragma once


//-------------------------------------------------------------------------------------------------
// Support Vector Machine
// ker=0, linear SVM    f(x) = w'*x + b 
// ker>0, nonlinear SVM f(x) = asv'*k(x,Xsv) + b
//-------------------------------------------------------------------------------------------------
class SVM
{
public:
	SVM(bool verbose = true);
	~SVM();

	// train linear SVM (LSVM) model
	// X[num*dim]
	// y[num] +1/-1 or 1/0
	// c trade-off parameter
	// opt 0:Hinge by FSMO; 1:Hinge-Huber by LBFGS; 2:Hinge-Square by LBFGS
	bool TrainLSVM(int num, int dim, float *X, int *y, float c, int opt = 0);
	// train LSVM with non-negative weights
	bool TrainLSVMnnw(int num, int dim, float *X, int *y, float c);
	
	// x[dim]
	float TestModel(float *x);

	// X[num*dim]
	// y[num]
	// return accuracy rate
	float TestModel(int num, float *X, int *y, const char *filename);

	float*GetLSVM(){ return wb; }
	void PrintLSVM();

	// filename, SVM model file
	bool LoadModel(const char *filename);
	bool SaveModel(const char *filename);

	static double FuncGrad(void *cls, int n, double *x, double *g);

private:
	int  FSMO(int n, float *X, int *y, float *x, float c, float tol);
	inline float KernelFunction(float *x1, float *x2);

	// training
	bool  verbose; // true default
	int   num; // training sample number
	int   dim; // feature dimension
	int   ker; // kernel type
	int   opt; // option for loss functions
	float arg; // kernel parameter
	float c;   // trade-off parameter
	float *X;  // pointer to training samples
	int   *y;  // pointer to labels

	// linear model
	float *wb; // linear model [w b]
	
	// nonlinear model
	int   nsv; // number of support vectors
	float *Xsv;// Xsv [nsv * dim]
	float *asv;// asv [nsv]
	float b;   // bias term
};
