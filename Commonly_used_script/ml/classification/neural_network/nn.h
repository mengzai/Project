#pragma once


//-------------------------------------------------------------------------------------------------
// Neural Network
//-------------------------------------------------------------------------------------------------
class NN
{
public:
	NN();
	~NN();

	bool Setup(int l, int *n);

	// cls, class number
	// X[num*dim]
	// y[num]
	// epochs, maximum training epochs
	int TrainModelBySGD(int num, float *X, int *y, int epochs = 50);

	// output response vector a
	// return class label, 0 ~ cls-1
	int TestModel(float *x);
	
	// X[num*dim]
	// y[num]
	// return accuracy rate
	float TestModel(int num, float *X, int *y);
	
	// filename, model file
	bool LoadModel(const char *filename);
	bool SaveModel(const char *filename);

	float* GetOutputs(){ return a[l-1]; }
	int  GetOutputNumber(){ return n[l-1]; }

	enum ACTIVATION_LOSS
	{
		SIGMOID_SQUARED,
		SIGMOID_SOFTMAX,
		ReLU_HINGE,
		ReLU_SQUAREDHINGE,
		ReLU_SMOOTHHINGE
	};
	ACTIVATION_LOSS opt; // option for activation function and loss function
	float lam; // regularization term lam/2*w'*w, default 0
	float dro; // dropout rate

private:
	void  Forward(float *x);
	float Loss(int y);
	void  Backward(float lr);

	void Remove();
	void Remove(int n, float** &x);

	int     l; // layer number
	int    *n; // neuron number for each layer
	float **W; // weight matrix between layers       W[k][nn[k+1]*nn[k]]
	float **b; // bias vector between layers         b[k][nn[k]]
	float **a; // activation vector for each layer   a[k][nn[k]], a[k+1][nn[k+1]]
	           //                                    a[k+1] = f(W[k]*a[k]+b[k])
	float **e; // error term vector for each layer   e[k][nn[k]], k>0

	int   num;
	float  *X;
	int    *y;
};
