#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include "sn.h"
#include "lbfgs.h"
#include "lbfgsb.h"
#pragma warning(disable:4996)


//-------------------------------------------------------------------------------------------------
SN::SN(bool verbose)
{
	this->verbose = verbose;
	num = 0;
	dim = 0;
	reg = 0;
	lam = 0;
	X = 0;
	y = 0;
	wb = 0;
}


//-------------------------------------------------------------------------------------------------
SN::~SN()
{
	if(wb!=0) delete[] wb;
}


//-------------------------------------------------------------------------------------------------
bool SN::TrainModel(int num, int dim, float *w, float *X, int *y, int reg, float lam, float *wb0)
{
	clock_t time = clock();

	if(reg==0) lam = 0;
	this->num = num;
	this->dim = dim;
	this->w = w;
	this->X = X;
	this->y = y;
	this->reg = reg;
	this->lam = lam;

	if(wb!=0) delete[] wb;
	wb = new float[dim+1];
	double *x = new double[dim+1];
	if(wb0==0) memset(x, 0, (dim+1)*sizeof(double));
	else{ for(int i=0; i<=dim; i++) x[i] = wb0[i]; }

	LBFGS lbfgs;
	lbfgs.verbose = verbose;
	if(reg==1) lbfgs.OptimizeOWLQN(this, SN::FuncGrad, dim+1, x, lam, 0, dim-1);
	else lbfgs.Optimize(this, SN::FuncGrad, dim+1, x);
	
	for(int i=0; i<=dim; i++) wb[i] = (float)x[i];
	delete[] x;

	time = clock()-time;
	printf("training time = %.3lf s\n\n", (double)time/CLOCKS_PER_SEC);

	return true;
}


//-------------------------------------------------------------------------------------------------
bool SN::TrainModelNNW(int num, int dim, float *w, float *X, int *y, int reg, float lam, float *wb0)
{
	if(reg==1) return false; // not supported yet for L1-norm positive weights
	
	clock_t time = clock();

	if(reg==0) lam = 0;
	this->num = num;
	this->dim = dim;
	this->w = w;
	this->X = X;
	this->y = y;
	this->reg = reg;
	this->lam = lam;

	if(wb!=0) delete[] wb;
	wb = new float[dim+1];
	double *x = new double[dim+1];
	double *l = new double[dim+1]; // lower bound
	double *u = new double[dim+1]; // upper bound
	int *nlu = new int[dim+1];     // bound type
	if(wb0==0) memset(x, 0, (dim+1)*sizeof(double));
	else{ for(int i=0; i<=dim; i++) x[i] = wb0[i]; }

	for(int i=0; i<dim; i++)
	{
		nlu[i] = 1;
		l[i] = u[i] = 0;
	}
	nlu[dim] = 0;
	l[dim] = u[dim] = 0;

	LBFGSB lbfgsb;
	if(!verbose) lbfgsb.iprint = -1;
	lbfgsb.Optimize(this, SN::FuncGrad, dim+1, x, nlu, l, u);
	
	for(int i=0; i<=dim; i++) wb[i] = (float)x[i];
	delete[] x;
	delete[] l;
	delete[] u;
	delete[] nlu;

	time = clock()-time;
	printf("training time = %.3lf s\n\n", (double)time/CLOCKS_PER_SEC);

	return true;
}


//-------------------------------------------------------------------------------------------------
float SN::TestModel(float *x)
{
	// step 1. p1 = w'*x+b
	float p1 = wb[dim]; // b
	for(int i=0; i<dim; i++) p1 += wb[i]*x[i]; // w'*x

	// step 2. p1 = 1/(1+exp(-p1)), a trick to avoid exp overflow
	if(p1>=0) p1 = 1/(1+exp(-p1));
	else
	{
		p1 = exp(p1);
		p1 = p1/(1+p1);
	}

	return p1;
}


//-------------------------------------------------------------------------------------------------
float SN::TestModel(int num, float *X, int *y, const char *filename)
{
	FILE *pfile = NULL;
	if(filename!=NULL) pfile = fopen(filename, "wt");

	int n00 = 0, n01 = 0, n10 = 0, n11 = 0;
	for(int i=0; i<num; i++)
	{
		float p = TestModel(X+i*dim);
		if(y[i]!=1 && p< 0.5f) n00++;
		if(y[i]!=1 && p>=0.5f) n01++;
		if(y[i]==1 && p< 0.5f) n10++;
		if(y[i]==1 && p>=0.5f) n11++;

		if(pfile!=NULL) fprintf(pfile, "%d\t%f\n", y[i], p);
	}

	if(pfile!=NULL) fclose(pfile);	
	
	printf("--------------------------------------------\n");
	printf("              真实正      真实负      准确率\n");
	printf("预测正  %12d%12d%11.2lf%%\n", n11, n01, 100.0*n11/(n11+n01));
	printf("预测负  %12d%12d%11.2lf%%\n", n10, n00, 100.0*n00/(n10+n00));
	printf("召回率  %11.2lf%%%11.2lf%%%11.2lf%%\n", 100.0*n11/(n11+n10), 100.0*n00/(n01+n00),
		100.0*(n00+n11)/num);
	printf("--------------------------------------------\n");
	
	return (float)(100.0*(n00+n11)/num);
}


//-------------------------------------------------------------------------------------------------
// calculate objective function
// SN    min                + 1/n*sum pi
// L1-SN min     lam*1'*|w| + 1/n*sum pi
// L2-SN min 0.5*lam*w'*w   + 1/n*sum pi
// calculate gradient vector
// SN              + 1/n*sum pi*(pi-1)*yi*[xi 1]'
// L1-SN lam*pg(w) + 1/n*sum pi*(pi-1)*yi*[xi 1]'
// L2-SN lam*w     + 1/n*sum pi*(pi-1)*yi*[xi 1]'
double SN::FuncGrad(void *cls, int n, double *x, double *g)
{
	SN *sn = (SN*)cls;
	int num = sn->num;
	int dim = sn->dim;
	int reg = sn->reg;
	float lam = sn->lam;
	float *w = sn->w; // training sample weights or NULL
	float *X = sn->X;
	int *y = sn->y;

	double f = 0;
	memset(g, 0, n*sizeof(double));
	
	float *xi = X;
	for(int i=0; i<num; i++)
	{
		// pi = 1/(1+exp(yi*(w'*xi+b)))	
		// step 1. pi = yi*(w'*xi+b)
		double pi = x[dim]; // b
		for(int j=0; j<dim; j++) pi += x[j]*xi[j]; // w'*xi
		if(y[i]!=1) pi = -pi;
		
		// step 2. pi = 1/(1+exp(pi)), a trick to avoid exp overflow
		if(pi<=0) pi = 1/(1+exp(pi));
		else
		{
			pi = exp(-pi);
			pi = pi/(1+pi);
		}
		if(w==NULL) f += pi; // f = sum pi
		else f += w[i]*pi;   // f = sum wi*pi

		// weights
		double gi = pi*(pi-1);
		if(y[i]!=1) gi = -gi;
		if(w!=NULL) gi *= w[i];
		for(int j=0; j<dim; j++) g[j] += gi*xi[j];
		g[dim] += gi; // bias

		xi += dim;
	}
	f /= num;
	for(int i=0; i<=dim; i++) g[i] /= num;

	if(reg==2)
	{
		// 0.5*lam*w'*w + ...
		double ww = 0;
		for(int i=0; i<dim; i++) ww += x[i]*x[i];
		f += 0.5*lam*ww;

		// lam*w + ...
		for(int i=0; i<dim; i++) g[i] += lam*x[i];
	}

	return f;
}


//-------------------------------------------------------------------------------------------------
void SN::PrintModel()
{
	for(int i=0; i<=dim; i++) printf(i==0 ? "wb = %11.4lf\n" : "%16.4lf\n", wb[i]);
}


//-------------------------------------------------------------------------------------------------
bool SN::LoadModel(const char *filename)
{
	FILE *pfile = fopen(filename, "rb");
	if(pfile==0) return false;
	
	dim = 0;
	if(wb!=0){ delete[] wb; wb = 0;	}

	fread(&dim, sizeof(int), 1, pfile);
	wb = new float[dim+1];
	fread(wb, sizeof(float), dim+1, pfile);
	fclose(pfile);
	
	return true;
}


//-------------------------------------------------------------------------------------------------
bool SN::SaveModel(const char *filename)
{
	if(dim==0 || wb==0) return false;

	FILE *pfile = fopen(filename, "wb");
	if(pfile==0) return false;
	
	fwrite(&dim, sizeof(int), 1, pfile);
	fwrite(wb, sizeof(float), dim+1, pfile);
	fclose(pfile);
	
	return true;
}
