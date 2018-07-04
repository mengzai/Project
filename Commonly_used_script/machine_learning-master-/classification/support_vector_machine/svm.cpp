#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include "svm.h"
#include "lbfgs.h"
#include "lbfgsb.h"
#pragma warning(disable:4996)


//-------------------------------------------------------------------------------------------------
SVM::SVM()
{
	num = 0; // training sample number
	dim = 0; // feature dimension
	ker = 0; // kernel type
	arg = 0; // kernel parameter
	c   = 0; // trade-off parameter
	X   = 0; // pointer to training samples
	y   = 0; // pointer to labels

	// linear model
	wb  = 0; // linear model [w b]
	
	// nonlinear model
	nsv = 0; // number of support vectors
	Xsv = 0; // Xsv [nsv * dim]
	asv = 0; // asv [nsv]
	b   = 0; // bias term
}


//-------------------------------------------------------------------------------------------------
SVM::~SVM()
{
	if(wb!=0)  delete[] wb;
	if(Xsv!=0) delete[] Xsv;
	if(asv!=0) delete[] asv;
}


//-------------------------------------------------------------------------------------------------
bool SVM::TrainLSVM(int num, int dim, float *X, int *y, float c, int opt)
{
	clock_t time = clock();

	this->ker = 0;
	this->num = num;
	this->dim = dim;
	this->opt = opt;
	this->X = X;
	this->y = y;
	this->c = c;

	if(wb!=0) delete[] wb;
	wb = new float[dim+1];

	if(opt==0) // default FSMO optimizer
	{
		const float tol = 0.001f;
		
		float *a = new float[num];
		char  *s = new char [num];

		FSMO(num, X, y, a, c, tol);

		// find support vectors on support planes
		int nsv = 0;
		for(int i=0; i<num; i++)
		{
			if(a[i]>tol)
			{
				nsv++;
				if(a[i]<c-tol) s[i] = 2;
				else s[i] = 1;
			}
			else s[i] = 0;
		}
		printf("Get %d support vectors\n", nsv);

		// weight vector w = sum y[i]*asv*x[i]
		for(int i=0; i<dim; i++)
		{
			float temp = 0;
			for(int j=0; j<num; j++) if(s[j]>0) temp += y[j]*a[j]*X[j*dim+i];
			wb[i] = temp;
		}

		// bias term b
		float b = 0;
		int  nb = 0;
		for(int i=0; i<num; i++)
		{
			if(s[i]!=2) continue;
			float temp = 0, *xi = X+i*dim;
			for(int j=0; j<dim; j++) temp += wb[j]*xi[j];
			b += y[i]-temp;
			nb++;
		}
		if(nb>0) b /= nb;
		wb[dim] = b;
	}
	else
	{
		double *x = new double[dim+1];
		memset(x, 0, (dim+1)*sizeof(double));

		LBFGS lbfgs;
		lbfgs.els = LBFGS::Armijo;
		lbfgs.Optimize(this, SVM::FuncGrad, dim+1, x);
	
		for(int i=0; i<=dim; i++) wb[i] = (float)x[i];
		delete[] x;
	}

	time = clock()-time;
	printf("training time = %.3lf s\n\n", (double)time/CLOCKS_PER_SEC);

	return true;
}


//-------------------------------------------------------------------------------------------------
// train LSVM with non-negative weights
bool SVM::TrainLSVMnnw(int num, int dim, float *X, int *y, float c)
{
	clock_t time = clock();

	this->ker = 0;
	this->num = num;
	this->dim = dim;
	this->opt = 2;
	this->X = X;
	this->y = y;
	this->c = c;

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
	lbfgsb.Optimize(this, SVM::FuncGrad, dim+1, x, nlu, l, u);
	
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
double SVM::FuncGrad(void *cls, int n, double *x, double *g)
{
	SVM *svm = (SVM*)cls;
	int  dim = svm->dim;
	int  num = svm->num;
	int  opt = svm->opt;
	float *X = svm->X;
	int   *y = svm->y;
	float  c = svm->c;

	double f = 0, f1 = 0;
	for(int i=0; i<dim; i++) f += x[i]*x[i];
	f *= 0.5;

	const double t0 = 0.10;
	memset(g, 0, n*sizeof(double));
	for(int i=0; i<num; i++)
	{
		const float *xi = X+i*dim;
		
		double t = x[dim];
		for(int j=0; j<dim; j++) t += x[j]*xi[j];
		int yi = y[i];
		if(yi==1) t = 1-t;
		else t = 1+t;

		// hinge loss max(0, 1-y*f(x)), max(0,t)
		if(opt==0 && t>0)
		{
			f1 += t;
			for(int j=0; j<dim; j++) g[j] -= yi*xi[j];
			g[dim] -= yi;
		}
		
		// huber hinge loss, t = 1-y*f(x)
		// if -t0<=t<=t0, (t-t0)^2/(4*t0)
		// if      t>=t0, t
		// otherwise,     0
		if(opt==1 && t>-t0)
		{
			if(t<t0)
			{
				t -= t0;
				f1 += t*t/(4*t0);
				t *= yi;
				for(int j=0; j<dim; j++) g[j] -= t*xi[j]; // (t-t0)*t'/(2*t0)
				g[dim] -= t;
			}
			else
			{
				f1 += t;
				for(int j=0; j<dim; j++) g[j] -= yi*xi[j]; // t'
				g[dim] -= yi;
			}
		}		

		// hinge-square loss 0.5*max(0, 1-y*f(x))^2, 0.5*max(0,t)^2
		if(opt==2 && t>0)
		{
			f1 += 0.5*t*t;
			t *= yi;
			for(int j=0; j<dim; j++) g[j] -= t*xi[j]; // t*t'
			g[dim] -= t;
		}
	}

	f += c*f1;
	for(int i=0; i<dim; i++) g[i] = x[i]+c*g[i];
	g[dim] *= c;

	return f;
}


//-------------------------------------------------------------------------------------------------
// Fast Sequential Minimal Optimization (FSMO)
// min  Q(x) = 0.5*x'*A*x - 1'*x,
// s.t. y'*x = 0, 0 <= xi <= c.
int SVM::FSMO(int n, float *X, int *y, float *x, float c, float tol)
{
	const int maxt = 1000000; // maximum iteration number
	const float minx = (float)1e-4, maxx = c-minx, eps = (float)1e-8; // x limits

	float *dQ = new float[n]; // dQ = y.*Q'(x)
	bool  *b1 = new bool [n];
	bool  *b2 = new bool [n];

	int row = (512<<20)/(n*sizeof(float)), uncached = 0; // 512MB
	if(row< 2) row = 2;
	if(row>=n) row = n;
	float *A = new float[row*n]; // cache rows of kernel matrix
	int *offset = 0; // offsets in cache for training samples, -1 means not cached
	int *index = 0, tail = 0; // row indexes for matrix A, tail the oldest row
	
	// case: memory is large enough to cache whole kernel matrix
	bool cached = row==n;
	if(cached)
	{
		printf("Kernel matrix can be fully cached\n");
		for(int i=0; i<n; i++)
		{
			float *xi = X+i*dim;
			A[i*n+i] = KernelFunction(xi, xi);
			for(int j=i+1; j<n; j++)
			{
				float *xj = X+j*dim;
				A[i*n+j] = A[j*n+i] = KernelFunction(xi, xj);
			}
		}
	}
	else
	{
		printf("Kernel matrix can be partially (%d rows, %.2f%%) cached\n", row, 100.0f*row/n);
		offset = new int[n];  // offsets in cache for training samples, -1 means not cached
		index = new int[row]; // row indexes for matrix A
		for(int i=0; i<n; i++) offset[i] = -1;
		for(int i=0; i<row; i++) index[i] = -1;
	}

	// initialize dQ, b1 and b2
	memset(x, 0, n*sizeof(float));
	for(int i=0; i<n; i++)
	{
		dQ[i] = (float)(-y[i]);
		b1[i] = y[i]<0; // yi*dx<0 -> yi<0 && xi<c || yi>0 && xi>0
		b2[i] = y[i]>0; // yi*dx>0 -> yi<0 && xi>0 || yi>0 && xi<c
	}

	// begin iterations of x and dQ
	printf("Start to optimize the problem by FSMO\n");
	int t;
	for(t=1; t<maxt; t++)
	{        
		// find two variables that produce a maximal reduction in Q
		float dQ1 = -1e10, dQ2 = 1e10;
		int i1, i2;
		for(int i=0; i<n; i++)
		{
			if(b1[i] && dQ1<dQ[i]){ dQ1 = dQ[i]; i1 = i; }
			if(b2[i] && dQ2>dQ[i]){ dQ2 = dQ[i]; i2 = i; }
		}

		// output optimization message
		float gap = dQ1-dQ2;
		if(t%1000==0 || gap<tol)
		{
			if(cached)
				printf("dQ1-dQ2 = %8.4f\n", gap); 
			else
			{
				printf("dQ1-dQ2 = %8.4f, uncached = %4d rows (%4.1f%%)\n", gap, uncached, 
					uncached/20.0f);
				uncached = 0;
			}
		}
		if(gap<tol) break;

		// optimize x(i1) and x(i2)
		int y1 = y[i1], y2 = y[i2];
		float x1 = x[i1], x2 = x[i2], a;
		if(cached)
			a = A[i1*n+i1]-2*A[i1*n+i2]+A[i2*n+i2];
		else
		{
			if(offset[i1]==-1) // if i1-row not cached
			{
				if(index[tail]>=0) offset[index[tail]] = -1; // uncache the previous row
				index[tail] = i1;
				offset[i1] = tail*n;
				if(++tail>=row) tail = 0;

				float *ai1 = A+offset[i1]; // cache the i1-row of kernel matrix
				float *xi1 = X+i1*dim, *xi = X;
				for(int i=0; i<n; i++)
				{
					ai1[i] = KernelFunction(xi1, xi);
					xi += dim;
				}
				uncached++;
			}

			if(offset[i2]==-1) // if i2-row not cached
			{
				if(index[tail]>=0) offset[index[tail]] = -1; // uncache the previous row
				index[tail] = i2;
				offset[i2] = tail*n;
				if(++tail>=row) tail = 0;

				float *ai2 = A+offset[i2]; // cache the i2-row of kernel matrix
				float *xi2 = X+i2*dim, *xi = X;
				for(int i=0; i<n; i++)
				{
					ai2[i] = KernelFunction(xi2, xi);
					xi += dim;
				}
				uncached++;
			}

			float *ai1 = A+offset[i1], *ai2 = A+offset[i2];
			a = ai1[i1]-2*ai1[i2]+ai2[i2];
		}
		if(a<eps) a = eps;

		// check the bounds and update the two variables
		float dx = (dQ1-dQ2)/a;
		float dx1 = y1>0 ? x1 : c-x1;
		float dx2 = y2>0 ? c-x2 : x2;
		if(dx>dx1) dx = dx1;
		if(dx>dx2) dx = dx2;
		if(y2>0) x2 += dx;
		else x2 -= dx;
		if(y1>0) x1 -= dx;
		else x1 += dx;
		x[i1] = x1;
		x[i2] = x2;

		// update gradient dQ and solution x
		float *ai1 = A + (cached ? i1*n : offset[i1]);
		float *ai2 = A + (cached ? i2*n : offset[i2]);
		for(int i=0; i<n; i++) dQ[i] += (ai2[i]-ai1[i])*dx;

		// update b1 and b2
		if(y1<0 && x1<maxx || y1>0 && x1>minx) b1[i1] = true;
		else b1[i1] = false;
		if(y1<0 && x1>minx || y1>0 && x1<maxx) b2[i1] = true;
		else b2[i1] = false;
		if(y2<0 && x2<maxx || y2>0 && x2>minx) b1[i2] = true;
		else b1[i2] = false;
		if(y2<0 && x2>minx || y2>0 && x2<maxx) b2[i2] = true;
		else b2[i2] = false;
	}

	delete[] b1;
	delete[] b2;
	delete[] dQ;
	delete[] A;
	if(offset!=0) delete[] offset;
	if(index !=0) delete[] index;

	printf("FSMO finished after %d iterations\n", t);
	return t;
}


//-------------------------------------------------------------------------------------------------
// kernel function
// 0 linear: x'*x2
// 1 polynomial: (x1'*x2)^arg
// 2 polynomial: (1+x1'*x2)^arg
// 3 Gaussian: exp(-||x1-x2||^2/(2*arg^2))
inline float SVM::KernelFunction(float *x1, float *x2)
{
	float f = 0;

	switch(ker)
	{
	case 0:
		for(int i=0; i<dim; i++) f += x1[i]*x2[i];
		break;

	case 1:
		for(int i=0; i<dim; i++) f += x1[i]*x2[i];
		f = pow(f, (int)arg);
		break;
	
	case 2:
		for(int i=0; i<dim; i++) f += x1[i]*x2[i];
		f = pow(1+f, (int)arg);
		break;

	default:
		for(int i=0; i<dim; i++)
		{
			float temp = x1[i]-x2[i];
			f += temp*temp;
		}
		f = exp(-f/(2*arg*arg));
		break;
	}

	return f;
}


//-------------------------------------------------------------------------------------------------
float SVM::TestModel(float *x)
{
	// f = w'*x+b
	float f = wb[dim]; // b
	for(int i=0; i<dim; i++) f += wb[i]*x[i]; // w'*x

	// output class probability
	const float a = 3.0f;
	if(f>=0) f = 1/(1+exp(-a*f));
	else
	{
		float temp = exp(a*f);
		f = temp/(1+temp);
	}

	return f;
}


//-------------------------------------------------------------------------------------------------
float SVM::TestModel(int num, float *X, int *y)
{
	int n00 = 0, n01 = 0, n10 = 0, n11 = 0;
	for(int i=0; i<num; i++)
	{
		float p = TestModel(X+i*dim);
		if(y[i]!=1 && p< 0.5f) n00++;
		if(y[i]!=1 && p>=0.5f) n01++;
		if(y[i]==1 && p< 0.5f) n10++;
		if(y[i]==1 && p>=0.5f) n11++;
	}
		
	printf("--------------------------------------------\n");
	printf("            真实正例    真实负例      准确率\n");
	printf("预测正例%12d%12d%11.2lf%%\n", n11, n01, 100.0*n11/(n11+n01));
	printf("预测负例%12d%12d%11.2lf%%\n", n10, n00, 100.0*n00/(n10+n00));
	printf("召回率  %11.2lf%%%11.2lf%%%11.2lf%%\n", 100.0*n11/(n11+n10), 100.0*n00/(n01+n00),
		100.0*(n00+n11)/num);
	printf("--------------------------------------------\n");

	return (float)(100.0*(n00+n11)/num);
}


//-------------------------------------------------------------------------------------------------
void SVM::PrintLSVM()
{
	for(int i=0; i<=dim; i++) printf(i==0 ? "wb = %11.4lf\n" : "%16.4lf\n", wb[i]);
}


//-------------------------------------------------------------------------------------------------
// filename, SVM model file
bool SVM::LoadModel(const char *filename)
{
	FILE *pfile = fopen(filename, "rb");
	if(pfile==0) return false;

	fread(&ker, sizeof(int), 1, pfile);
	// linear SVM model
	if(ker==0)
	{
		dim = 0;
		if(wb!=0){ delete[] wb; wb = 0;	}

		fread(&dim, sizeof(int), 1, pfile);
		wb = new float[dim+1];
		fread(wb, sizeof(float), dim+1, pfile);
	}
	// sparse nonlinear SVM model
	else
	{
	}
	fclose(pfile);
	
	return true;
}


//-------------------------------------------------------------------------------------------------
bool SVM::SaveModel(const char *filename)
{
	// linear SVM model
	if(ker==0)
	{
		if(dim==0 || wb==0) return false;

		FILE *pfile = fopen(filename, "wb");
		if(pfile==0) return false;

		fwrite(&ker, sizeof(int), 1, pfile);
		fwrite(&dim, sizeof(int), 1, pfile);
		fwrite(wb, sizeof(float), dim+1, pfile);
		fclose(pfile);
	}
	// sparse nonlinear SVM model
	else
	{
	}
	
	return true;
}
