#pragma once


//-------------------------------------------------------------------------------------------------
class LBFGS
{
public:
    typedef double (*PtrFuncGrad)(void *cls, int n, double *x, double *g);

    int Optimize(void *cls, PtrFuncGrad ptr, int n, double *x);

	// Orthant-Wise Limited-memory Quasi-Newton, L1-regularized model
	// c*|xi| + model, where i = ix0, ix0+1, ..., ix1, and 0<=ix0 && ix1<n
	int OptimizeOWLQN(void *cls, PtrFuncGrad ptr, int n, double *x, double c, int ix0, int ix1);

	// parameters for LBFGS
	static bool   verbose; // true default
	static int    maxiter; // maximum number of iterations
	static int    dirtrun; // the k most recent directions/sizes truncated
	static double minstep; // minimum allowable step length
	static enum LineSearch{Armijo, StrongWolfe} els; // line search method, default StrongWolfe
	static int    maxls;   // maximum number of line search iterations
	static double c1;      // sufficient decrease parameter
	static double c2;      // curvature parameter

private:
	bool LbfgsAdd();
	void LbfgsProd();
	
	bool LineSearchArmijo(void *cls, PtrFuncGrad ptr);
	bool LineSearchStrongWolfe(void *cls, PtrFuncGrad ptr);
	bool LineSearchQwlqn(void *cls, PtrFuncGrad ptr, double c, int ix0, int ix1);

	double OwlqnL1norm(double c, int ix0, int ix1);
	void OwlqnPseudoGrad(double c, int ix0, int ix1);
	
	int     n;  // x's dimension
	double  f;  // function value
	double *x0; // current solution
	double *x1; // previous solution
	double *g0; // current gradient
	double *g1; // previous gradient
	double *pg; // owlqn pseudo gradient
	double *d;  // search direction
	double  t;  // step size
	double gd;  // directional derivative g'*d

	double *s;  // the k most recent t*d
	double *y;  // the k most recent g0-g1
	double *ys; // the k most recent y'*s
	double *al; // alpha used by LbfgsProd
	double  h;  // scale of initial Hessian approximation
	int     ia; // index of lbfgs end
	int     na; // number of added s, y and ys
	int     ls; // line search iterations
};
