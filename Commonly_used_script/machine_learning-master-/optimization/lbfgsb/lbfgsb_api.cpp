#include "lbfgsb_api.h"
#include "lbfgsb.h"


//-------------------------------------------------------------------------------------------------
int    LBFGSB::maxiter = 10000;
int    LBFGSB::m       = 5;
int    LBFGSB::iprint  = 1;   // 1 have output at every iteration
double LBFGSB::factr   = 1e7; // 1e12 for low accuracy
                              // 1e7 for moderate accuracy
                              // 1e1 for extremely high accuracy
double LBFGSB::pgtol   = 1e-5;


//-------------------------------------------------------------------------------------------------
// n - dimension of the problem
int LBFGSB::Optimize(void *cls, PtrFuncGrad ptr, int n, double *x, int *nlu, double *l, double *u)
{
	int ret = 0;

	// check parameters
	if(ptr==0 || n<=0 || x==0) return ret;

	double  f = 0;             // function value
	double *g = new double[n]; // gradient values
	double*wa = new double[(2*m+5)*n+11*m*m+8*m]; // working array
	int  *iwa = new int[3*n];  // working array
    double dsave[29]; // information
    int    isave[44]; // integer information
    int    lsave[4];  // logical information

    // We start the iteration by initializing task
	int task = START;
	int csave = 0;

	int iter = 0;
	while(iter<maxiter)
	{
		// minimization rountine
		iter = iter+1;

		// this is the call to the L-BFGS-B code
		setulb(&n, &m, x, l, u, nlu, &f, g, &factr, &pgtol, wa, iwa, &task, &iprint, &csave, lsave, 
			isave, dsave);

		if(IS_FG(task))
		{
			// the minimization routine has returned to request the function f and gradient g values
			// at the current x
			f = ptr(cls, n, x, g);
		}

		// if task is neither FG nor NEW_X we terminate execution
		else if(task!=NEW_X) break;
	}

	delete[] g;
	delete[] wa;
	delete[] iwa;

	return ret;
}
