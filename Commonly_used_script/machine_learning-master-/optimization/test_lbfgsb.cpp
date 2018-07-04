#include <stdio.h>
#include "lbfgsb.h"


//-------------------------------------------------------------------------------------------------
double FuncGrad(void *cls, int n, double *x, double *g)
{
    // Computing 2nd power
    double d1 = x[0]-1.;
    double f = d1*d1*.25;
    for(int i=2; i<=n; ++i)
	{
        // Computing 2nd power
        double d2 = x[i-2];
        d1 = x[i-1]-d2*d2;
        f += d1*d1;
    }
    f *= 4.;

    // Compute gradient g for the sample problem
    // Computing 2nd power
    d1 = x[0];
    double t1 = x[1]-d1*d1;
    g[0] = (x[0]-1.)*2.-x[0]*16.*t1;
    for(int i=2; i<n; ++i)
	{
        // Computing 2nd power
        double t2 = t1;
        d1 = x[i-1];
        t1 = x[i]-d1*d1;
        g[i-1] = t2*8.-x[i-1]*16.*t1;
    }
    g[n-1] = t1*8.;

	//printf("f = %.4lf%\n", f);
	return f;
}


//-------------------------------------------------------------------------------------------------
int main()
{
	const int n = 25;
	double x[n] = {0}, l[n], u[n];
	int nlu[n];

	// First set bounds on the odd-numbered variables
    for(int i=1; i<=n; i+=2)
	{
        nlu[i-1] = 2;
        l[i-1] = 1.;
        u[i-1] = 100.;
    }
    // Next set bounds on the even-numbered variables
    for(int i=2; i<=n; i+=2)
	{
        nlu[i-1] = 2;
        l[i-1] = -100.;
        u[i-1] =  100.;
    }
    
	// We now define the starting point
    for(int i=1; i<=n; ++i) x[i-1] = 3.;

    printf("     Solving sample problem (Rosenbrock test fcn).\n");
    printf("      (f = 0.0 at the optimal solution.)\n");

	LBFGSB lbfgsb;
	lbfgsb.Optimize(0, FuncGrad, n, x, nlu, l, u);

	for(int i=0; i<n; i++) printf("\t%.4f\n", x[i]);

	return 0;
}
