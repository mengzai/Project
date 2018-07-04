#pragma once
#undef  EXPORT
#ifdef  WIN32
#define EXPORT __declspec(dllexport) 
#else
#define EXPORT
#endif


//-------------------------------------------------------------------------------------------------
class EXPORT LBFGSB
{
public:
	static int    maxiter; // maximum number of iterations
	static int    m;       // the number m of limited memory corrections stored
	static int    iprint;  // it controls the frequency and type of output generated

	// the tolerances in the stopping criteria
	// (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch
	// max{|proj g_i | i = 1, ..., n} <= pgtol
	static double factr;
	static double pgtol;

	// nlu - the type of bounds imposed on the variables
	//       nlu(i)=0 if x(i) is unbounded
	//              1 if x(i) has only a lower bound
	//              2 if x(i) has both lower and upper bounds
	//              3 if x(i) has only an upper bound
	typedef double (*PtrFuncGrad)(void *cls, int n, double *x, double *g);
	int Optimize(void *cls, PtrFuncGrad ptr, int n, double *x, int *nlu, double *l, double *u);
};
