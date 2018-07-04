#include <math.h>
#include <float.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include "heapsort.h"
#include "gbrt.h"
#pragma warning(disable:4996)


//-------------------------------------------------------------------------------------------------
GBRT::GBRT()
{
	loss = R_SQUARE;
	max_depth = 4;
	min_samples_leaf = 1;
	learning_rate = 0.1f;
	huber_quantile = 0.9f;
	ytr = NULL;
	
	d = 0;
	M = 0;
	F0 = 0;
	Fm = NULL;
	Qm = 0;
}


//-------------------------------------------------------------------------------------------------
GBRT::~GBRT()
{
	Release();
}


//-------------------------------------------------------------------------------------------------
void GBRT::Release()
{
	if(Fm!=NULL)
	{
		for(int m=0; m<M; m++)
		{
			if(Fm[m]!=NULL)
			{
				std::stack<treenode*> s;
				s.push(Fm[m]);
		
				while(!s.empty())
				{
					treenode *tn = s.top();
					s.pop();
					if(tn->tn0!=NULL) s.push(tn->tn0);
					if(tn->tn1!=NULL) s.push(tn->tn1);
					delete tn;
				}
			}
		}

		delete[] Fm;
		Fm = NULL;
	}
}


//-------------------------------------------------------------------------------------------------
void GBRT::SetParam(LOSS loss, int max_depth, int min_samples_leaf, float learning_rate, float
					huber_quantile)
{
	this->loss = loss;
	this->max_depth = max_depth;
	this->min_samples_leaf = min_samples_leaf;
	this->learning_rate = learning_rate;
	this->huber_quantile = huber_quantile;
}


//-------------------------------------------------------------------------------------------------
void GBRT::Train(int n, int d, float *X, float *y, int tmax)
{
	Release();
	this->d = d;
	ytr = y;

	// sort training samples by each attribute
	int *idn = SortFeatures(n, X); // idn[dxn]

	// negative gradient and residuals
	int size = n; // -gradient
	if(loss>R_SQUARE) size = 2*n; // -gradient, residual
	float *ym = new float[size];

	// F0(x) and ym
	M = 0;
	TrainF0(n, y, ym);

	// Fm(x) and ym
	int stop = 0; // convergence test
	float err_old = 0;
	Fm = new treenode*[tmax];
	for(int m=0; m<tmax; m++) Fm[m] = NULL;
	for(int m=0; m<tmax; m++)
	{
		M++;
		Fm[m] = new treenode();
		memset(Fm[m], 0, sizeof(treenode));

		TrainFm(m, n, X, ym, idn);

		float err = Test(n, X, y, ym);
		printf("estimator F%03d(x), %s = %.4f\n", M, loss>=C_LOGISTIC?"error":"loss(F(x)-y)", err);	

		// convergence test
		if(err==0) break;
		if(fabs(err-err_old)<1e-4*err){ if(++stop>=3) break; }
		err_old = err;
		
		// update ym: negative gradient & residuals
		UpdateYm(n, y, ym);
	}

	delete[] idn;
	delete[] ym;
}


//-------------------------------------------------------------------------------------------------
float GBRT::Test(int n, float *X, float *y, float *f)
{
	double err = 0;
	
	for(int i=0; i<n; i++)
	{
		double fi = F0;
		
		float *xi = X+i*d;
		for(int m=0; m<M; m++)
		{
			// f += Fm;
			treenode *tn = Fm[m];
			while(tn->tn0!=NULL)
			{
				if(xi[tn->att]<tn->thr) tn = tn->tn0;
				else tn = tn->tn1;
			}
			fi += learning_rate*tn->y;
		}
		if(f!=NULL) f[i] = (float)fi;

		if(loss==R_SQUARE) err += (fi-y[i])*(fi-y[i]);
		else if(loss==R_ABSOLUTE) err += fabs(fi-y[i]);
		else if(loss==R_HUBER) err += fabs(fi-y[i]);
		else if(loss>=C_LOGISTIC){ if(y[i]*fi<=0) err++; }
		
		//printf("[%f %f]\n", y[i], fi);
	}
	err /= n;

	//printf("error = %.3lf%%\n", 100.0*err);
	return (float)err;
}


//-------------------------------------------------------------------------------------------------
void GBRT::TrainF0(int n, float *y, float *ym)
{	
	if(loss==R_SQUARE)
	{
		// F0 = mean(yi)
		double f = 0;
		for(int i=0; i<n; i++) f += y[i];
		f /= n;
		F0 = (float)f;
		
		// gradient vector and residuals = y-F0
		for(int i=0; i<n; i++) ym[i] = y[i]-F0; // negative gradient = residuals
	}
	else if(loss==R_ABSOLUTE)
	{
		// F0 = median(yi)
		F0 = GetMedian(n, y);

		// gradient vector and residuals = y-F0
		for(int i=0; i<n; i++)
		{
			float ri = y[i]-F0;
			ym[n+i] = ri; // residuals
			if(ri>1e-4) ym[i] = 1; // negative gradient
			else if(ri<-1e-4) ym[i] = -1;
			else ym[i] = 0;
		}
	}
	else if(loss==R_HUBER)
	{
		// F0 = median(yi)
		F0 = GetMedian(n, y);

		// residuals = y-F0;
		for(int i=0; i<n; i++) ym[n+i] = y[i]-F0; // residuals

		// negative gradient
		Qm = GetQuantile(n, ym+n);
		for(int i=0; i<n; i++)
		{
			float ri = ym[n+i];
			if(ri>=Qm) ym[i] = Qm; // negative gradient
			else if(ri<=-Qm) ym[i] = -Qm;
			else ym[i] = ri;
		}
	}
	else if(loss==C_LOGISTIC)
	{
		// F0 = 0.5*log((1+ymean)/(1-ymean))
		double f = 0;
		for(int i=0; i<n; i++) f += y[i];
		f /= n;
		F0 = (float)(0.5*log((1+f)/(1-f)));
		
		// negative gradient
		for(int i=0; i<n; i++) ym[i] = 2*y[i]/(1+exp(2*y[i]*F0));
	}
	else if(loss==C_RELU)
	{
		// F0 = mean(yi)
		double f = 0;
		for(int i=0; i<n; i++) f += y[i];
		f /= n;
		//F0 = (float)f;
		if(f>=0) F0 = 1;
		else F0 = -1;
		
		// negative gradient of max(0, -y*F(x))
		for(int i=0; i<n; i++)
		{
			ym[i] = y[i]*F0<1e-4 ? y[i] : 0; // negative gradient
			ym[n+i] = F0; // current value
		}
	}
}


//-------------------------------------------------------------------------------------------------
void GBRT::TrainFm(int m, int n, float *X, float *y, int *idn)
{
	// build decision tree
	std::stack<treenode*> s;
	std::stack<bool*> b;

	bool *mask = new bool[n];
	for(int i=0; i<n; i++) mask[i] = true;
	s.push(Fm[m]);
	b.push(mask);

	while(!s.empty())
	{
		treenode *tn = s.top();
		s.pop();

		bool *mask = b.top();
		b.pop();

		if(SplitNode(n, X, y, idn, mask, tn))
		{
			treenode *tn0 = new treenode();
			treenode *tn1 = new treenode();
			memset(tn0, 0, sizeof(treenode));
			memset(tn1, 0, sizeof(treenode));

			// split the current set into two subsets
			bool *mask0 = new bool[n];
			bool *mask1 = new bool[n];
			float *xatt = X+tn->att, thr = tn->thr;
			for(int i=0; i<n; i++)
			{
				mask0[i] = mask1[i] = false;
				if(mask[i])
				{
					if(*xatt<thr) mask0[i] = true;
					else mask1[i] = true;
				}
				xatt += d;
			}

			// add left child and right child
			tn0->dep = tn1->dep = tn->dep+1;
			tn->tn0 = tn0, tn->tn1 = tn1;
			s.push(tn0);
			s.push(tn1);
			b.push(mask0);
			b.push(mask1);
		}

		// release mask
		delete[] mask;
	}
}


//-------------------------------------------------------------------------------------------------
int* GBRT::SortFeatures(int n, float *X)
{
	int* idn = new int[d*n];

	// sort training samples by each attribute
	float *x = new float[n];
	for(int k=0; k<d; k++)
	{
		float *xk = X+k;
		for(int i=0; i<n; i++, xk+=d) x[i] = *xk;

		HeapSort<float> hs;
		hs.SortAsce(n, x, idn+k*n);
	}
	delete[] x;

	return idn;
}


//-------------------------------------------------------------------------------------------------
// splite node by least square criteria
bool GBRT::SplitNode(int n, float *X, float *y, int *idn, bool *mask, treenode *tn)
{
	// if it's a leaf node, just return
	int m = 0; // the number of samples in this node
	double ytn = 0, impu = 0, sy2 = 0;
	for(int i=0; i<n; i++)
	{ 
		if(mask[i])
		{
			ytn += y[i];
			sy2 += y[i]*y[i]; // sum(yi^2)
			m++;
		}
	}
	ytn /= m;
	impu = sy2-m*ytn*ytn;
	if(tn->dep>=max_depth || m<2*min_samples_leaf || impu<1e-8)
	{
		// it's a leaf node
		if(loss==R_SQUARE) tn->y = (float)ytn;
		else if(loss==R_ABSOLUTE) tn->y = GetMedian(n, y+n, mask, m);
		else if(loss==R_HUBER) tn->y = GetHuber(n, y+n, mask, m);
		else if(loss==C_LOGISTIC) tn->y = GetLogistic(n, y, mask, m);
		else if(loss==C_RELU) tn->y = GetReLU(n, y+n, mask, m);
		//printf("tn->y = %f\n", tn->y);
		return false;
	}
	
	// find samples in this node
	int *idm = new int[d*m];
	for(int k=0; k<d; k++)
	{
		int *ik = idn+k*n;
		for(int i=0, j=k*m; i<n; i++)
		{
			if(mask[ik[i]]) idm[j++] = ik[i];
		}
	}

	// find attribute and threshold to minimize the impurity
	bool ret = true;
	double impu_min = 1e10;
	double *s = new double[m]; // for accuracy, need double
	for(int k=0; k<d; k++) // for each attribute
	{
		int *ik = idm+k*m, iki; // corresponding indexes

		iki = ik[0];
		s[0] = y[iki];
		for(int i=1; i<m; i++)
		{
			iki = ik[i];
			s[i] = s[i-1]+y[iki];
		}
		
		for(int i=min_samples_leaf-1, i1=m-min_samples_leaf; i<i1; i++) // for each threshold
		{
			// threshold must satisfy "x_prev < x_next"
			if(X[ik[i]*d+k]<X[ik[i+1]*d+k])
			{
				double f0 = s[i]/(i+1.0);
				double f1 = (s[m-1]-s[i])/(m-i-1.0);
				double impu = sy2-(i+1)*f0*f0-(m-i-1)*f1*f1;

				//printf("impu = %f\n", impu);
				if(impu_min>impu)
				{
					impu_min = impu;
					tn->att = k;
					tn->thr = (X[ik[i]*d+k]+X[ik[i+1]*d+k])/2;
				}
			}
		}
	}
	printf("att = %d, thr = %f, impu_min = %f\n", tn->att, tn->thr, impu_min);

	delete[] idm;
	delete[] s;

	return ret;
}


//-------------------------------------------------------------------------------------------------
float GBRT::GetMedian(int n, float *y, bool *b, int m)
{
	float *ym = new float[m];
	for(int i=0, j=0; i<n; i++){ if(b[i]) ym[j++] = y[i]; }
	
	float f = GetMedian(m, ym);	
	
	delete[] ym;
	return f;
}


//-------------------------------------------------------------------------------------------------
float GBRT::GetMedian(int n, float *y)
{
	const int k = (n%2==0) ? n/2+1 : (n+1)/2;
	float *ak = new float[k];
	int   *bk = new int[k];
	
	HeapSort<float> hs;
	hs.SortMaxK(n, y, k, ak, bk);

	// median(yi)
	float f;
	if(n%2==0) f = (ak[k-2]+ak[k-1])/2;
	else f = ak[k-1];
	
	delete[] ak;
	delete[] bk;
	return f;
}


//-------------------------------------------------------------------------------------------------
float GBRT::GetQuantile(int n, float *y)
{
	const int k = (int)((n-1)*(1-huber_quantile)+1);
	float *a  = new float[n];
	float *ak = new float[k];
	int   *bk = new int[k];

	for(int i=0; i<n; i++) a[i] = y[i]>=0 ? y[i] : -y[i]; // a = |y|
	
	HeapSort<float> hs;
	hs.SortMaxK(n, a, k, ak, bk);	
	
	float f = ak[k-1];

	delete[] a;
	delete[] ak;
	delete[] bk;
	if(f<0.001f) f = 0.001f;
	return f;
}


//-------------------------------------------------------------------------------------------------
float GBRT::GetHuber(int n, float *y, bool *b, int m)
{
	float *ym = new float[m];
	for(int i=0, j=0; i<n; i++){ if(b[i]) ym[j++] = y[i]; }	
	float median = GetMedian(m, ym);
	delete[] ym;
	
	double offset = 0;
	for(int i=0; i<n; i++)
	{
		if(b[i])
		{
			float d = y[i]-median;
			if(d>=0) offset += (Qm<=d ? Qm : d);
			else offset -= (Qm<=-d ? Qm : -d);
		}
	}
	float f = (float)(median+offset/m);

	return f;
}


//-------------------------------------------------------------------------------------------------
float GBRT::GetLogistic(int n, float *y, bool *b, int m)
{
	double f0 = 0, f1 = 0;

	for(int i=0; i<n; i++)
	{
		if(b[i])
		{
			f0 += y[i];
			f1 += fabs(y[i])*(2-fabs(y[i]));
		}
	}

	return (float)(f0/f1);
}


//-------------------------------------------------------------------------------------------------
float GBRT::GetReLU(int n, float *fy, bool *b, int m)
{
	float *f = fy, *y = ytr;
	float f0 = -FLT_MAX, f1 = FLT_MAX;
	for(int i=0; i<n; i++)
	{
		if(b[i])
		{
			if(y[i]>0){ if(f1>f[i]) f1 = f[i]; } // min from positive
			else{ if(f0<f[i]) f0 = f[i]; }       // max from negative
		}
	}
	
	if(f0==-FLT_MAX) return  1024-f1; // only positives
	if(f1== FLT_MAX) return -1024-f0; // only negatives
	if(f0<f1) return -(f1+f0)/2; // separable

	int m0 = 0, m1 = 0;
	for(int i=0; i<n; i++)
	{
		if(b[i] && f0<=f[i] && f[i]<=f1)
		{
			if(y[i]<0) m0++;
			else m1++;
		}
	}
	m = m0+m1;

	// copy overlapped region
	float *fm = new float[m];
	float *ym = new float[m];
	int   *im = new   int[m];
	for(int i=0, j=0; i<n; i++)
	{
		if(b[i] && f0<=f[i] && f[i]<=f1)
		{
			fm[j] = f[i];
			ym[j] = y[i];
			j++;
		}
	}
	HeapSort<float> hs;
	hs.SortAsce(m, fm, im);
	
	// search optimal response
	/*float ret = 0, min_loss = FLT_MAX;
	for(int i=0; i<m-1; i++)
	{
		float gamma = 0;
		if(i==-1) gamma = fm[0]-1;
		else if(i+1<m) gamma = (fm[i]+fm[i+1])/2;
		else gamma = fm[i]+1;
		
		float loss = 0;
		for(int j=0; j<m; j++)
		{
			float temp = -ym[j]*(fm[i]+gamma);
			if(temp>0) loss += temp;
		}
		if(min_loss>loss) min_loss = loss, ret = gamma;
	}
	printf("ret = %f\n", ret);*/

	float ret = 0;
	for(int i=0, i0=0, i1=0; i<m-1; i++)
	{
		if(ym[im[i]]<0) i0++;
		else i1++;
		if(i0==m1-i1){ ret = -(fm[i]+fm[i+1])/2; break; }
	}
		
	//for(int i=0; i<m; i++) printf("%f,%c\t", fm[i], ym[i]==1?'+':'-');
	//printf("ret = %f\n", ret);
	//printf("gamma = %f, min_loss = %f\n", ret, min_loss);

/*	int isep = 0, dsep = m;
	for(int i=0, i0=0, i1=0; i<m; i++)
	{
		if(ym[im[i]]<0) i0++;
		else i1++;
		
		if(i==0 || i+1==m || fm[i]<fm[i+1])
		{
			int d = abs(m1-i0-i1);
			if(dsep>d) isep = i, dsep = d;
		}
	}
	//for(int i=0; i<m; i++) printf("%f ", fm[i]);
	//printf("\n");
	printf("m = %d, isep = %d, dsep = %d \n", m, isep, dsep);

	float ret = 0;
	if(isep==0)
	{
		if(isep+1<m && fm[isep]<fm[isep+1]) ret = -(fm[isep]+fm[isep+1])/2; // non-separable
		else ret = -fm[0]-fabs(fm[0]); // all to negative
	}
	else if(isep+1==m) ret = -fm[m-1]+fabs(fm[m-1]); // all to positive
	else ret = -(fm[isep]+fm[isep+1])/2; // non-separable
*/
	delete[] fm;
	delete[] ym;
	delete[] im;

	return ret;
}


//-------------------------------------------------------------------------------------------------
void GBRT::UpdateYm(int n, float *y, float *ym)
{
	switch(loss)
	{
	case R_SQUARE:
		for(int i=0; i<n; i++) ym[i] = y[i]-ym[i]; // negative gradient ym = y-F(x)
		break;

	case R_ABSOLUTE:
		for(int i=0; i<n; i++)
		{
			float ri = y[i]-ym[i];
			ym[n+i] = ri; // residuals
			if(ri>1e-4) ym[i] = 1; // negative gradient
			else if(ri<-1e-4) ym[i] = -1;
			else ym[i] = 0;
		}
		break;

	case R_HUBER:
		for(int i=0; i<n; i++) ym[n+i] = y[i]-ym[i]; // residuals

		Qm = GetQuantile(n, ym+n);
		for(int i=0; i<n; i++)
		{
			float ri = ym[n+i];
			if(ri>=Qm) ym[i] = Qm; // negative gradient
			else if(ri<=-Qm) ym[i] = -Qm;
			else ym[i] = ri;
		}
		break;

	case C_LOGISTIC:
		for(int i=0; i<n; i++) ym[i] = 2*y[i]/(1+exp(2*y[i]*ym[i])); // negative gradient
		break;

	case C_RELU: // max(0, -y*F(x))
		for(int i=0; i<n; i++)
		{
			ym[n+i] = ym[i]; // current value
			ym[i] = y[i]*ym[i]<1e-4 ? y[i] : 0; // negative gradient
			//if(fabs(ym[i])<1e-4) ym[i] = 0.5*y[i];
			//else if(y[i]*ym[i]>0) ym[i] = 0;
			//else ym[i] = y[i];
			//printf("updateYm [y = %f, -g = %f, f = %f]\n", y[i], y[i]*ym[i], ym[n+i]);
		}
		break;
	}
}


//-------------------------------------------------------------------------------------------------
bool GBRT::LoadModel(const char *filename)
{
	Release();

	FILE *pfile = fopen(filename, "rb");
	if(pfile==0) return false;

	bool ret = true;
	fread(&loss, sizeof(int), 1, pfile); // loss
	fread(&d, sizeof(int), 1, pfile); // dimension
	fread(&M, sizeof(int), 1, pfile); // estimators
	fread(&learning_rate, sizeof(float), 1, pfile); // learning rate
	fread(&F0, sizeof(float), 1, pfile); // initial estimator
	Fm = new treenode*[M];
	for(int i=0; i<M; i++)
	{
		// load regression tree model
		Fm[i] = NULL;
		std::stack<treenode*> s;
		while(ret)
		{
			// load tree node
			treenode temp;
			memset(&temp, 0, sizeof(treenode));
			if(fread(&(temp.dep), sizeof(int), 1, pfile)!=1) break;
			if(fread(&(temp.att), sizeof(int), 1, pfile)!=1) break;
			if(fread(&(temp.thr), sizeof(float), 1, pfile)!=1) break;
			if(fread(&(temp.y),   sizeof(float), 1, pfile)!=1) break;

			if(temp.dep==0)
			{
				if(Fm[i]==NULL)
				{
					Fm[i] = new treenode();
					memcpy(Fm[i], &temp, sizeof(treenode)); // root node
					s.push(Fm[i]);
				}
				else
				{
					// complete to load this regression tree model
					long offset = -(long)(2*sizeof(int)+2*sizeof(float));
					fseek(pfile, offset, SEEK_CUR);
					break;
				}
			}
			else if(!s.empty())
			{
				// find its parent
				treenode *tn = s.top();
				while(!s.empty() && temp.dep<=tn->dep)
				{
					s.pop();
					tn = s.top();
				}

				// attach a child
				if(temp.dep==tn->dep+1 && (tn->tn0==NULL || tn->tn1==NULL))
				{
					treenode *temptn = new treenode();
					memcpy(temptn, &temp, sizeof(treenode));
					if(tn->tn1==NULL) tn->tn1 = temptn;
					else tn->tn0 = temptn;
					s.push(temptn);
				}
				else{ printf("error: no parent node\n"); ret = false; }
			}
			else{ printf("error: no root node\n"); ret = false; }
		}
		
		if(Fm[i]==NULL){ printf("error: empty tree\n"); ret = false; }
	}

	fclose(pfile);
	return ret;
}


//-------------------------------------------------------------------------------------------------
bool GBRT::SaveModel(const char *filename)
{
	if(Fm==NULL) return false;

	FILE *pfile = fopen(filename, "wb");
	if(pfile==0) return false;

	fwrite(&loss, sizeof(int), 1, pfile); // loss
	fwrite(&d, sizeof(int), 1, pfile); // dimension
	fwrite(&M, sizeof(int), 1, pfile); // estimators
	fwrite(&learning_rate, sizeof(float), 1, pfile); // learning rate
	fwrite(&F0, sizeof(float), 1, pfile); // initial estimator
	for(int i=0; i<M; i++)
	{
		// save regression tree model
		std::stack<treenode*> s;
		s.push(Fm[i]);
		while(!s.empty())
		{
			// save tree node
			treenode *tn = s.top();
			fwrite(&(tn->dep), sizeof(int), 1, pfile);
			fwrite(&(tn->att), sizeof(int), 1, pfile);
			fwrite(&(tn->thr), sizeof(float), 1, pfile);
			fwrite(&(tn->y),   sizeof(float), 1, pfile);

			// push its two childs
			s.pop();
			if(tn->tn0!=NULL) s.push(tn->tn0);
			if(tn->tn1!=NULL) s.push(tn->tn1);
		}
	}

	fclose(pfile);
	return true;
}
