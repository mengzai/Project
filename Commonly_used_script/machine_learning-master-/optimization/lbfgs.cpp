#include "lbfgs.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>


//-------------------------------------------------------------------------------------------------
bool   LBFGS::verbose = true;
int    LBFGS::maxiter = 1000000;
int    LBFGS::dirtrun = 10;
double LBFGS::minstep = 1e-9;
LBFGS::LineSearch LBFGS::els = StrongWolfe;
int    LBFGS::maxls   = 21;
double LBFGS::c1      = 0.0001;
double LBFGS::c2      = 0.9;


//-------------------------------------------------------------------------------------------------
int LBFGS::Optimize(void *cls, PtrFuncGrad ptr, int n, double *x)
{
	int ret = 0;

	// check parameters
	if(ptr==0 || n<=0 || x==0) return ret;

	this->n = n;        // x's dimension
	x0 = new double[n]; // current solution
	x1 = new double[n]; // previous solution
	g0 = new double[n]; // current gradient
	g1 = new double[n]; // previous gradient
	d  = new double[n]; // search direction
	pg = 0;             // owlqn pseudo gradient

	s  = new double[dirtrun*n]; // the k most recent t*d
	y  = new double[dirtrun*n]; // the k most recent g0-g1
	ys = new double[dirtrun];   // the k most recent y'*s
	al = new double[dirtrun];   // alpha used by LbfgsProd
	
	memcpy(x0, x, n*sizeof(double));
	f = ptr(cls, n, x0, g0);
	ls = 1;
	
	int iter = 0;
	while(iter<maxiter)
	{
		if(iter==0)
		{
			memset(s, 0, dirtrun*n*sizeof(double));
			memset(y, 0, dirtrun*n*sizeof(double));
			memset(ys,0, dirtrun*sizeof(double));
			
			t = 0;
			for(int i=0; i<n; i++)
			{
				d[i] = -g0[i]; // d = -g
				t += fabs(g0[i]);
				//t += g0[i]*g0[i];
			}
			t = t<=1 ? 1 : 1/t; // t = min(1,1/sum(abs(g)));
			//t = t<=1 ? 1 : 1/sqrt(t); // t = min(1,1/sqrt(g'*g));
			h = 1;
			ia = 0;
			na = 0;
		}
		else
		{
			LbfgsAdd();
			LbfgsProd();
			
			t = 1; // Newton step
		}
		iter++;

		// directional derivative g'*d
		gd = 0;
		for(int i=0; i<n; i++) gd += g0[i]*d[i];
		if(gd>-minstep) break;

		// line search for step size
		if(els==Armijo)
		{
			if(!LineSearchArmijo(cls, ptr)) break;
		}
		else if(els==StrongWolfe)
		{
			if(!LineSearchStrongWolfe(cls, ptr)) break;
		}

		// output iteration information
		if(verbose) printf("iter = %3d, f = %.8lf, t = %.4lf, gd = %.8lf\n", iter, f, t, gd);
	}
	if(verbose) printf("\nL-BFGS iterations = %d, function evalutions = %d\n", iter, ls);
	
	memcpy(x, x0, n*sizeof(double));

	delete[] x0;
	delete[] x1;
	delete[] g0;
	delete[] g1;
	delete[] d;
	delete[] s;
	delete[] y;
	delete[] ys;
	delete[] al;

	return ret;
}


//-------------------------------------------------------------------------------------------------
// Orthant-Wise Limited-memory Quasi-Newton, L1-regularized model
// c*|xi| + model, where i = ix0, ix0+1, ..., ix1, and 0<=ix0 && ix1<n
int LBFGS::OptimizeOWLQN(void *cls, PtrFuncGrad ptr, int n, double *x, double c, int ix0, int ix1)
{
	int ret = 0;

	// check parameters
	if(ptr==0 || n<=0 || x==0 || !(0<=ix0 && ix0<=ix1 && ix1<n)) return ret;

	this->n = n;        // x's dimension
	x0 = new double[n]; // current solution
	x1 = new double[n]; // previous solution
	g0 = new double[n]; // current gradient
	g1 = new double[n]; // previous gradient
	pg = new double[n]; // owlqn pseudo gradient
	d  = new double[n]; // search direction

	s  = new double[dirtrun*n]; // the k most recent t*d
	y  = new double[dirtrun*n]; // the k most recent g0-g1
	ys = new double[dirtrun];   // the k most recent y'*s
	al = new double[dirtrun];   // alpha used by LbfgsProd
	
	memcpy(x0, x, n*sizeof(double));
	f = ptr(cls, n, x0, g0);
	f += OwlqnL1norm(c, ix0, ix1); // add L1-regularization
	OwlqnPseudoGrad(c, ix0, ix1);  // pseudo gradient
	ls = 1;
	
	int iter = 0;
	while(iter<maxiter)
	{
		if(iter==0)
		{
			memset(s, 0, dirtrun*n*sizeof(double));
			memset(y, 0, dirtrun*n*sizeof(double));
			memset(ys,0, dirtrun*sizeof(double));
			
			t = 0;
			for(int i=0; i<n; i++)
			{
				d[i] = -pg[i]; // d = -pg
				t += fabs(pg[i]);
			}
			t = t<=1 ? 1 : 1/t; // t = min(1,1/sum(abs(pg)))
			h = 1;
			ia = 0;
			na = 0;
		}
		else
		{
            // update direction
			if(LbfgsAdd()) LbfgsProd();

			// constrain search direction to match the sign pattern of pg
			for(int i=ix0; i<=ix1; i++)
			{
				if(d[i]*pg[i]>=0) d[i] = 0;
			}
			
			t = 1; // Newton step
		}
		iter++;

		// line search for step size
		if(!LineSearchQwlqn(cls, ptr, c, ix0, ix1)) break;

		// output iteration information
		if(verbose) printf("iter = %3d, f = %.8lf, t = %.4lf, gd = %.8lf\n", iter, f, t, gd);

		// test convergence
		if(gd>-minstep) break;
	}
	if(verbose) printf("\nOWL-QN iterations = %d, function evalutions = %d\n", iter, ls);
	
	memcpy(x, x0, n*sizeof(double));

	delete[] x0;
	delete[] x1;
	delete[] g0;
	delete[] g1;
	delete[] pg;
	delete[] d;
	delete[] s;
	delete[] y;
	delete[] ys;
	delete[] al;

	return ret;
}


//-------------------------------------------------------------------------------------------------
bool LBFGS::LbfgsAdd()
{
	double *si = s+ia*n;
	double *yi = y+ia*n;
	double ysi = 0;
    
    // if ysi<1e-10, skip update
    for(int i=0; i<n; i++) ysi += (g0[i]-g1[i])*(t*d[i]); // ys = y'*s
    if(ysi<1e-10) return false;

	// add the current s and y
	h = 0;
	for(int i=0; i<n; i++)
	{
		si[i] = t*d[i];
		yi[i] = g0[i]-g1[i];
		//ysi += yi[i]*si[i];
		h += yi[i]*yi[i]; // h = yi'*yi
	}
	ys[ia] = ysi;

	// update scale of initial Hessian approximation
	h = ysi/h; // h = ysi/(yi'*yi)

	// update ia and na
	ia++;
	if(ia==dirtrun) ia = 0;
	if(na<dirtrun) na++;
    return true;
}


//-------------------------------------------------------------------------------------------------
void LBFGS::LbfgsProd()
{
	// d = -g
	double *temp = pg==0 ? g0 : pg;
	for(int j=0; j<n; j++) d[j] = -temp[j];

	// backward loop, --
	for(int j=0; j<na; j++)
	{
		int i = ia-1-j;
		if(i<0) i += dirtrun;

		// ali = si'*d/ysi
		double ali = 0, *si = s+i*n;
		for(int k=0; k<n; k++) ali += si[k]*d[k];
		ali /= ys[i];
		al[i] = ali;

		// d = d-ali*yi
		double *yi = y+i*n;
		for(int k=0; k<n; k++) d[k] -= ali*yi[k];
	}

	// d = h*d, multiply by initial Hessian
	for(int j=0; j<n; j++) d[j] *= h;

	// forward loop, ++
	for(int j=0; j<na; j++)
	{
		int i = na<dirtrun ? j : ia+j;
		if(i>=dirtrun) i -= dirtrun;

		// bei = yi'*d/ysi
		double bei = 0, *yi = y+i*n;
		for(int k=0; k<n; k++) bei += yi[k]*d[k];
		bei /= ys[i];

		// d = d+(ali-bei)*si
		bei = al[i]-bei;
		double *si = s+i*n;
		for(int k=0; k<n; k++) d[k] += bei*si[k];

		i--;
		if(i<0) i+= dirtrun;
	}
}


//-------------------------------------------------------------------------------------------------
bool LBFGS::LineSearchArmijo(void *cls, PtrFuncGrad ptr)
{
	// swap x0 and x1, g0 and g1
	double *temp = x0;
	x0   = x1;
	x1   = temp;
	temp = g0;
	g0   = g1;
	g1   = temp;

	// search a step length satisfying strong Wolfe conditions
	const double dec = 0.5;
	int lsi = 0;
	while(lsi<maxls)
	{
		// x0 = x1+t*d
		for(int i=0; i<n; i++) x0[i] = x1[i]+t*d[i];
		
		// update f(x0), g(x0) and dg0
		double f0 = ptr(cls, n, x0, g0), gd0 = 0;		
		for(int i=0; i<n; i++) gd0 += g0[i]*d[i];
		lsi++;

		// check Armijo condition
		if(f0>f+c1*t*gd) t *= dec;
		else
		{
			f = f0;
			break;
		}
	}
	ls += lsi;

	bool ret = lsi<maxls;
	if(verbose && !ret) printf("LineSearchArmijo failed\n");
	return ret;
}


//-------------------------------------------------------------------------------------------------
bool LBFGS::LineSearchStrongWolfe(void *cls, PtrFuncGrad ptr)
{
	// swap x0 and x1, g0 and g1
	double *temp = x0;
	x0   = x1;
	x1   = temp;
	temp = g0;
	g0   = g1;
	g1   = temp;

	// search a step length satisfying strong Wolfe conditions
	const double dec = 0.5, inc = 2.1;
	int lsi = 0;
	while(lsi<maxls)
	{
		// x0 = x1+t*d
		for(int i=0; i<n; i++) x0[i] = x1[i]+t*d[i];
		
		// update f(x0), g(x0) and dg0
		double f0 = ptr(cls, n, x0, g0), gd0 = 0;	
		for(int i=0; i<n; i++) gd0 += g0[i]*d[i];
		lsi++;

		// check strong Wolfe conditions
		if(f0>f+c1*t*gd) t *= dec;
		else
		{
			if(gd0<c2*gd) t *= inc; // gd0<-c2*|gd|
			else
			{
				if(gd0>-c2*gd) t *= dec; // gd0>c2*|gd|
				else
				{
					f = f0;
					break;
				}
			}
		}
	}
	ls += lsi;

	bool ret = lsi<maxls;
	if(verbose && !ret) printf("LineSearchStrongWolfe failed\n");
	return ret;
}


//-------------------------------------------------------------------------------------------------
bool LBFGS::LineSearchQwlqn(void *cls, PtrFuncGrad ptr, double c, int ix0, int ix1)
{
	// swap x0 and x1, g0 and g1
	double *temp = x0;
	x0   = x1;
	x1   = temp;
	temp = g0;
	g0   = g1;
	g1   = temp;

	// search a step length satisfying strong Wolfe conditions
	bool ret = true;
	const double dec = 0.5;
	int lsi = 0;
	while(lsi<maxls)
	{
		// x0 = x1+t*d
		for(int i=0; i<n; i++) x0[i] = x1[i]+t*d[i];

		// project the search point onto the orthant of the previous point
		for(int i=ix0; i<=ix1; i++)
		{
			if(x1[i]==0)
			{
				if(x0[i]*(-pg[i])<0) x0[i] = 0;
			}
			else if(x0[i]*x1[i]<0) x0[i] = 0;
		}
		
		// update f(x0), g(x0) and dg0
		double f0 = ptr(cls, n, x0, g0);
		f0 += OwlqnL1norm(c, ix0, ix1); // add L1-regularization
		lsi++;
		
		// directional derivative
		gd = 0;
		for(int i=0; i<n; i++) gd += (x0[i]-x1[i])*pg[i];

		// check Armijo condition
		if(f0>f+c1*t*gd) t *= dec;
		else
		{
			f = f0;
			OwlqnPseudoGrad(c, ix0, ix1);  // pseudo gradient
			break;
		}
	}
	if(lsi==maxls) ret = false;
	ls += lsi;

	return ret;
}


//-------------------------------------------------------------------------------------------------
double LBFGS::OwlqnL1norm(double c, int ix0, int ix1)
{
	double l1norm = 0;
	for(int i=ix0; i<=ix1; i++) l1norm += fabs(x0[i]);
	l1norm *= c;
	return l1norm;
}


//-------------------------------------------------------------------------------------------------
void LBFGS::OwlqnPseudoGrad(double c, int ix0, int ix1)
{
	memcpy(pg, g0, n*sizeof(double));
	
	for(int i=ix0; i<=ix1; i++)
	{
		if(x0[i]<0) pg[i] -= c;
		else if(x0[i]>0) pg[i] += c;
		else
		{
			if(pg[i]<-c) pg[i] += c;
			else if(pg[i]>c) pg[i] -= c;
			else pg[i] = 0;
		}
	}
}
