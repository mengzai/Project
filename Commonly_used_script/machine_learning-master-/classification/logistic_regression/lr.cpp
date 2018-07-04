#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include "lr.h"
#include "lbfgs.h"
#include "lbfgsb.h"
#pragma warning(disable:4996)


//-------------------------------------------------------------------------------------------------
LR::LR(bool verbose)
{
	this->verbose = verbose;
	num = 0;
	dim = 0;
	reg = 0;
	lam = 0;
	w = 0;
	X = 0;
	y = 0;
	wb = 0;
}


//-------------------------------------------------------------------------------------------------
LR::~LR()
{
	if(wb!=0) delete[] wb;
}


//-------------------------------------------------------------------------------------------------
bool LR::TrainModel(int num, int dim, float *w, float *X, int *y, int reg, float lam)
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
	memset(x, 0, (dim+1)*sizeof(double));

	LBFGS lbfgs;
	lbfgs.verbose = verbose;
	if(reg==1) lbfgs.OptimizeOWLQN(this, LR::FuncGrad, dim+1, x, lam, 0, dim-1);
	else lbfgs.Optimize(this, LR::FuncGrad, dim+1, x);
	
	for(int i=0; i<=dim; i++) wb[i] = (float)x[i];
	delete[] x;

	time = clock()-time;
	if(verbose) printf("training time = %.3lf s\n\n", (double)time/CLOCKS_PER_SEC);

	return true;
}


//-------------------------------------------------------------------------------------------------
bool LR::TrainModelNNW(int num, int dim, float *w, float *X, int *y, int reg, float lam)
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
	memset(x, 0, (dim+1)*sizeof(double));

	for(int i=0; i<dim; i++)
	{
		nlu[i] = 1;
		l[i] = u[i] = 0;
	}
	nlu[dim] = 0;
	l[dim] = u[dim] = 0;

	LBFGSB lbfgsb;
	if(!verbose) lbfgsb.iprint = -1;
	lbfgsb.Optimize(this, LR::FuncGrad, dim+1, x, nlu, l, u);
	
	for(int i=0; i<=dim; i++) wb[i] = (float)x[i];
	delete[] x;
	delete[] l;
	delete[] u;
	delete[] nlu;

	time = clock()-time;
	if(verbose) printf("training time = %.3lf s\n\n", (double)time/CLOCKS_PER_SEC);

	return true;
}


//-------------------------------------------------------------------------------------------------
float LR::TestModel(float *x, float &f)
{
	// step 1. p1 = w'*x+b
	float p1 = wb[dim]; // b
	for(int i=0; i<dim; i++) p1 += wb[i]*x[i]; // w'*x
	f = p1;

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
float LR::TestModel(int num, float *X, int *y, const char *filename)
{
	FILE *pfile = NULL;
	if(filename!=NULL) pfile = fopen(filename, "wt");
	
	int n00 = 0, n01 = 0, n10 = 0, n11 = 0;
	for(int i=0; i<num; i++)
	{
		float f = 0;
		float p = TestModel(X+i*dim, f);
		if(y[i]!=1 && p< 0.5f) n00++;
		if(y[i]!=1 && p>=0.5f) n01++;
		if(y[i]==1 && p< 0.5f) n10++;
		if(y[i]==1 && p>=0.5f) n11++;
		
		if(pfile!=NULL) fprintf(pfile, "%d\t%f\n", y[i], f);
	}
	
	if(pfile!=NULL) fclose(pfile);
	
	printf("--------------------------------------------\n");
	printf("            真实正例    真实负例      准确率\n");
	printf("预测正例%12d%12d%11.2lf%%\n", n11, n01, 100.0*n11/(n11+n01));
	printf("预测负例%12d%12d%11.2lf%%\n", n10, n00, 100.0*n00/(n10+n00));
	printf("召回率  %11.2lf%%%11.2lf%%%11.2lf%%\n", 100.0*n11/(n11+n10), 100.0*n00/(n01+n00), 100.0*(n00+n11)/num);
	printf("--------------------------------------------\n");

	return (float)(100.0*(n00+n11)/num);
}


//-------------------------------------------------------------------------------------------------
// calculate objective function
// LR    min                - 1/n*sum log(pi)
// L1-LR min     lam*1'*|w| - 1/n*sum log(pi)
// L2-LR min 0.5*lam*w'*w   - 1/n*sum log(pi)
// calculate gradient vector
// LR              - 1/n*sum (1-pi)*yi*[xi 1]'
// L1-LR lam*pg(w) - 1/n*sum (1-pi)*yi*[xi 1]'
// L2-LR lam*w     - 1/n*sum (1-pi)*yi*[xi 1]'
double LR::FuncGrad(void *cls, int n, double *x, double *g)
{
	LR *lr = (LR*)cls;
	int num = lr->num;
	int dim = lr->dim;
	int reg = lr->reg;
	float lam = lr->lam;
	float *w = lr->w; // training sample weights or NULL
	float *X = lr->X;
	int *y = lr->y;

	double f = 0;
	memset(g, 0, n*sizeof(double));

	float *xi = X;
	for(int i=0; i<num; i++)
	{
		// pi = 1/(1+exp(-yi*(w'*xi+b)))
		// step 1. pi = yi*(w'*xi+b)
		double pi = x[dim]; // b
		for(int j=0; j<dim; j++) pi += x[j]*xi[j]; // w'*xi
		if(y[i]!=1) pi = -pi;
		
		// step 2. pi = 1/(1+exp(-pi)), a trick to avoid exp overflow
		if(pi>=0) pi = 1/(1+exp(-pi));
		else
		{
			pi = exp(pi);
			pi = pi/(1+pi);
		}
		if(w==NULL) f += log(pi); // f = sum log(pi)
		else f += w[i]*log(pi);   // f = sum wi*log(pi)

		// weights
		double gi = 1-pi;
		if(y[i]!=1) gi = -gi;
		if(w!=NULL) gi *= w[i];
		for(int j=0; j<dim; j++) g[j] += gi*xi[j];
		g[dim] += gi; // bias

		xi += dim;
	}
	f /= -num;
	for(int i=0; i<=dim; i++) g[i] /= -num;

	// if reg==1, OWL-QN will do regularization part later
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
// filename, LR model file
bool LR::LoadModel(const char *filename)
{
	if(strcmp(filename+strlen(filename)-3, "txt")==0)
	{
		FILE *pfile = fopen(filename, "rt");
		if(pfile==0) return false;
	
		dim = 0;
		if(wb!=0){ delete[] wb; wb = 0;	}

		fscanf(pfile, "%d", &dim);
		wb = new float[dim+1];
		for(int i=0; i<=dim; i++) fscanf(pfile, "%f", wb+i);
		fclose(pfile);
	}
	else
	{
		FILE *pfile = fopen(filename, "rb");
		if(pfile==0) return false;
	
		dim = 0;
		if(wb!=0){ delete[] wb; wb = 0;	}

		fread(&dim, sizeof(int), 1, pfile);
		wb = new float[dim+1];
		fread(wb, sizeof(float), dim+1, pfile);
		fclose(pfile);
	}
	return true;
}


//-------------------------------------------------------------------------------------------------
void LR::PrintModel()
{
	for(int i=0; i<=dim; i++) printf(i==0 ? "wb = %11.4lf\n" : "%16.4lf\n", wb[i]);
}


//-------------------------------------------------------------------------------------------------
bool LR::SaveModel(const char *filename)
{
	if(dim==0 || wb==0) return false;

	if(strcmp(filename+strlen(filename)-3, "txt")==0)
	{
		FILE *pfile = fopen(filename, "wt");
		if(pfile==0) return false;
	
		fprintf(pfile, "%d\n", dim);
		for(int i=0; i<=dim; i++) fprintf(pfile, "%f\n", wb[i]);
		fclose(pfile);
	}
	else
	{
		FILE *pfile = fopen(filename, "wb");
		if(pfile==0) return false;
	
		fwrite(&dim, sizeof(int), 1, pfile);
		fwrite(wb, sizeof(float), dim+1, pfile);
		fclose(pfile);
	}
	return true;
}
