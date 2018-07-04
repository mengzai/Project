#include <math.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "shuffle.h"
#include "nn.h"


//-------------------------------------------------------------------------------------------------
NN::NN()
{
	l = 0;
	n = 0;
	W = 0;
	b = 0;
	a = 0;
	e = 0;
	lam = 0;
	opt = ReLU_HINGE;
	dro = 0;
}


//-------------------------------------------------------------------------------------------------
NN::~NN()
{
	Remove();
}


//-------------------------------------------------------------------------------------------------
bool NN::Setup(int l, int *n)
{
	if(l<1) return false;
	for(int k=0; k<l; k++){ if(n[k]<1) return false; }

	Remove();

	this->l = l;
	this->n = new int[l];
	memcpy(this->n, n, l*sizeof(int));
	
	W = new float*[l];
	b = new float*[l];
	a = new float*[l];
	e = new float*[l];
	
	for(int k=0; k<l; k++)
	{
		if(k+1<l)
		{
			int nk = n[k], nk1 = n[k+1];
			W[k] = new float[nk1*nk];
			b[k] = new float[nk1];
		}
		else
		{
			W[k] = 0;
			b[k] = 0;
		}
		a[k] = new float[n[k]];
		if(k==0) e[k] = 0;
		else e[k] = new float[n[k]];
	}

	// random initialization for W and b
	//unsigned int seed = (unsigned int)time(0); printf("%d\n", seed);
	unsigned int seed = 1433402319;
	srand(seed); // seed the random number generator
	for(int k=0; k<l-1; k++)
	{
		int nk1 = n[k+1], nk1nk = nk1*n[k];
		float *Wk = W[k], *bk = b[k];
		for(int i=0; i<nk1nk; i++)
		{
			double x = (double)rand()/(double)RAND_MAX; // 0.0 ~ 1.0
			Wk[i] = (float)(0.01*(2*x-1));
		}
		for(int i=0; i<nk1; i++) bk[i] = 0; // bias term
	}

	printf("%d-layer NN: ", l);
	for(int k=0; k<l; k++) printf(k+1<l ? "%d => " : "%d\n", n[k]);
	return true;
}


//-------------------------------------------------------------------------------------------------
int NN::TrainModelBySGD(int num, float *X, int *y, int epochs)
{
	float lr = 0.01f, decay = 0.90f;
	const int dim = n[0];

	Shuffle shuffle(num);
	int *idx = shuffle.RandomShuffle();
	for(int t=0; t<epochs; t++)
	{
		printf("\rbegin the %2d-th epoch of training\n", t);
		dro = 1;
		for(int i=0; i<num; i++)
		{
			//int ii = idx[i];
			Forward(X+i*dim);

			float lossi = Loss(y[i]);

			if(lossi>0) Backward(lr);
		}
		
		dro = 2;
		for(int i=0; i<num; i++)
		{
			//int ii = idx[i];
			Forward(X+i*dim);

			float lossi = Loss(y[i]);

			if(lossi>0) Backward(lr);
		}
		
		TestModel(num, X, y);		

		lr *= decay;
		if(lr<0.00005f) lr = 0.00005f;
	}
	dro = 0;

	return 0;
}


//-------------------------------------------------------------------------------------------------
void NN::Forward(float *x)
{
	memcpy(a[0], x, n[0]*sizeof(float));
	for(int k=0; k<l-1; k++)
	{
		// layer k -> layer k+1
		// zk1 = Wk*ak+bk
		// ak1 = f(zk1)

		const int nk = n[k], nk1 = n[k+1];
		float *Wk = W[k], *ak = a[k], *bk = b[k], *ak1 = a[k+1];

		for(int i=0; i<nk1; i++)
		{
			if(k==0 && (dro==1 && 2*i<nk1 || dro==2 && 2*i>=nk1)) continue;
			float *wki = Wk+i*nk;
			
			float zk1 = bk[i];		
			for(int j=0; j<nk; j++) zk1 += wki[j]*ak[j];

			switch(opt)
			{
			case SIGMOID_SQUARED:
				// f(z) = 1/(1+exp(-z))
				if(zk1>=0) ak1[i] = 1/(1+exp(-zk1));
				else
				{
					zk1 = exp(zk1);
					ak1[i] = zk1/(1+zk1);
				}
				break;

			case SIGMOID_SOFTMAX:
				if(k<l-2)
				{
					// f(z) = 1/(1+exp(-z))
					if(zk1>=0) ak1[i] = 1/(1+exp(-zk1));
					else
					{
						zk1 = exp(zk1);
						ak1[i] = zk1/(1+zk1);
					}
				}
				else ak1[i] = zk1;
				break;

			case ReLU_HINGE:
			case ReLU_SQUAREDHINGE:
			case ReLU_SMOOTHHINGE:
				if(k<l-2)
				{
					// ReLU f(z) = max(0,z)
					if(zk1>0) ak1[i] = zk1;
					else ak1[i] = 0;
				}
				else ak1[i] = zk1;
				break;
			}
		}
	}
}


//-------------------------------------------------------------------------------------------------
float NN::Loss(int y)
{
	int    no = n[l-1];
	float *ao = a[l-1]; // output units
	float *eo = e[l-1]; // errors

	// e(z) = loss'(a)*f'(z)
	float loss = 0;
	switch(opt)
	{
	float sum;
	case SIGMOID_SQUARED:
		// Sigmoid & Squared loss
		for(int i=0; i<no; i++)
		{
			float di = no==1 ? ao[i]-y : (y==i ? ao[i]-1 : ao[i]);
			loss += 0.5f*di*di;
			eo[i] = di*ao[i]*(1-ao[i]); // f'(z) = f(z)*(1-f(z))
		}
		break;

	case SIGMOID_SOFTMAX:
		if(no==1)
		{
			// loss = log(1+exp(-y*ao))
			sum = y==1 ? -ao[0] : ao[0]; // sum = -y*ao
			if(sum<=0)
			{
				sum = exp(sum);
				loss += log(1+sum);
				sum = sum/(1+sum);
			}
			else
			{
				sum = exp(-sum);
				loss -= log(sum/(1+sum));
				sum = 1/(1+sum);
			}
			eo[0] = y==1 ? -sum : sum;
		}
		else
		{
			// a trick to avoid exp overflow
			sum = ao[0]; // sum <- max{ zi }
			for(int i=1; i<no; i++) if(sum<ao[i]) sum = ao[i];
			for(int i=0; i<no; i++) ao[i] -= sum; // zi <- zi - maxz
		
			// Softmax-loss = log(sum exp(zi))-zy
			sum = 0;
			loss = -ao[y];
			for(int i=0; i<no; i++)
			{
				ao[i] = exp(ao[i]);
				sum += ao[i];
			}
			for(int i=0; i<no; i++) ao[i] /= sum;
			loss += log(sum);

			for(int i=0; i<no; i++)
			{
				if(y==i) eo[i] = ao[i]-1; // f'(z) = ao[i]-I{y==i}
				else eo[i] = ao[i];
			}
		}
		break;

	case ReLU_HINGE:
		// ReLU & Hinge-loss
		for(int i=0; i<no; i++)
		{
			if(no==1 && y==1 || no>1 && y==i)
			{
				if(ao[i]<1)
				{
					loss += 1-ao[i];
					eo[i] = -1;
				}
				else eo[i] = 0;
			}
			else
			{
				if(ao[i]>0)
				{
					loss += ao[i];
					eo[i] = 1;
				}
				else eo[i] = 0;
				//if(ao[i]>-1)
				//{
				//	loss += 1+ao[i];
				//	eo[i] = 1;
				//}
				//else eo[i] = 0;
			}
		}
		break;

	case ReLU_SQUAREDHINGE:
		// ReLU & squared Hinge-loss
		for(int i=0; i<no; i++)
		{
			if(no==1 && y==1 || no>1 && y==i)
			{
				if(ao[i]<1)
				{
					float temp = ao[i]-1;
					loss += temp*temp/2;
					eo[i] = temp;
				}
				else eo[i] = 0;
			}
			else
			{
				if(ao[i]>0)
				{
					loss += ao[i]*ao[i]/2;
					eo[i] = ao[i];
				}
				else eo[i] = 0;
				//if(ao[i]>-1)
				//{
				//	float temp = ao[i]+1;
				//	loss += temp*temp/2;
				//	eo[i] = temp;
				//}
				//else eo[i] = 0;
			}
		}
		break;
		
	case ReLU_SMOOTHHINGE:
		// ReLU & smooth Hinge loss
		for(int i=0; i<no; i++)
		{
			if(no==1 && y==1 || no>1 && y==i)
			{
				if(ao[i]<0) // linear
				{
					loss += 0.5f-ao[i];
					eo[i] = -1;
				}
				else if(ao[i]<1) // squared
				{
					float temp = ao[i]-1;
					loss += temp*temp/2;
					eo[i] = temp;
				}
				else eo[i] = 0;
			}
			else
			{
				if(ao[i]>1) // linear
				{
					loss += ao[i]-0.5;
					eo[i] = 1;
				}
				else if(ao[i]>0) // squared
				{
					loss += ao[i]*ao[i]/2;
					eo[i] = ao[i];
				}
				else eo[i] = 0;
			}
		}		
		break;
	}
	
	// regularization term used for weight decay or maximal margin
	if(loss>0 && lam>0)
	{
		float reg = 0, *wi = W[l-2];
		for(int i=0, ni=no*n[l-2]; i<ni; i++) reg += wi[i]*wi[i];
		loss += lam*reg/2;
	}
	
	return loss;
}


//-------------------------------------------------------------------------------------------------
void NN::Backward(float lr)
{
	for(int k=l-2; k>=0; k--)
	{
		// layer k+1 -> layer k
		// ek = (Wk'*ek1).f'(zk)
		
		const int nk = n[k], nk1 = n[k+1];
		float *ek1 = e[k+1], *ak = a[k];
		if(k>0)
		{
			float *ek = e[k], *Wk = W[k];
			for(int i=0; i<nk; i++)
			{
				float eki = 0, *wki = Wk+i;
				for(int j=0; j<nk1; j++, wki+=nk) eki += (*wki)*ek1[j];
				
				switch(opt)
				{
				case SIGMOID_SQUARED:
				case SIGMOID_SOFTMAX:
					ek[i] = eki*ak[i]*(1-ak[i]); // f'(z) = f(z)*(1-f(z))
					break;
				
				case ReLU_HINGE:
				case ReLU_SQUAREDHINGE:
				case ReLU_SMOOTHHINGE:
					// ReLU f(z) = max(0,z)
					if(ak[i]>0) ek[i] = eki; // f'(z) = 1
					else ek[i] = 0; // f'(z) = 0
					break;
				}
			}
		}
	}

	for(int k=0; k<l-1; k++)
	{
		// partial derivatives
		// dWk = ek1*ak'
		// dbk = ek1
		const int nk = n[k], nk1 = n[k+1];
		float *ek1 = e[k+1], *ak = a[k], *Wk = W[k], *bk = b[k];
		for(int i=0; i<nk1; i++)
		{if(k==0 && (dro==1 && 2*i<nk1 || dro==2 && 2*i>=nk1)) continue;
			float *Wki = Wk+i*nk, ek1i = ek1[i];
			//for(int j=0; j<nk; j++) Wki[j] -= lr*ek1i*ak[j]; // W <- W-lr*dW
			for(int j=0; j<nk; j++)
			{
				if(k==l-2 && lam>0) Wki[j] -= lr*(ek1i*ak[j]+lam*Wki[j]); // W <- W-lr*dW
				else Wki[j] -= lr*ek1i*ak[j]; // W <- W-lr*dW
			}
			bk[i] -= lr*ek1i; // b <- b-lr*db
		}
	}
}


//-------------------------------------------------------------------------------------------------
void NN::Remove()
{
	Remove(l, W);
	Remove(l, b);
	Remove(l, a);
	Remove(l, e);
	
	l = 0;
	if(n!=0){ delete[] n; n = 0; }
}


//-------------------------------------------------------------------------------------------------
void NN::Remove(int n, float** &x)
{
	if(x!=0)
	{
		for(int i=0; i<n; i++){ if(x[i]!=0) delete[] x[i]; }
		delete[] x;
		x = 0;
	}
}


//-------------------------------------------------------------------------------------------------
int NN::TestModel(float *x)
{
	Forward(x);

	int no = n[l-1];    // number of output units
	float *ao = a[l-1]; // output units
	if(opt==SIGMOID_SOFTMAX) // class probabilities for softmax loss
	{
		if(no==1) // binary classification
		{
			if(ao[0]>0) ao[0] = 1/(1+exp(-ao[0]));
			else
			{
				float f = exp(ao[0]);
				ao[0] = f/(1+f);
			}
		}
		else // multiclass classification
		{
			// a trick to avoid exp overflow
			float f = ao[0]; // sum <- max{ zi }
			for(int i=1; i<no; i++) if(f<ao[i]) f = ao[i];
			for(int i=0; i<no; i++) ao[i] -= f; // ai <- ai - maxa
			
			f = 0;
			for(int i=0; i<no; i++)
			{
				ao[i] = exp(ao[i]);
				f += ao[i];
			}
			for(int i=0; i<no; i++) ao[i] /= f; // sum ai = 1
		}
	}	

	int idx = 0;
	if(no==1) idx = ao[0]>0.5f ? 1 : 0; // binary classification
	else
	{
		for(int i=1; i<no; i++) if(ao[idx]<ao[i]) idx = i;
	}
	return idx;
}


//-------------------------------------------------------------------------------------------------
float NN::TestModel(int num, float *X, int *y)
{
	int n00 = 0, n01 = 0, n10 = 0, n11 = 0, out = n[l-1];
	int dim = n[0], cor = 0;
	for(int i=0; i<num; i++)
	{
		int yi = TestModel(X+i*dim);
		if(y[i]==yi) cor++;
		if(out==1)
		{
			if(y[i]!=1 && yi!=1) n00++;
			if(y[i]!=1 && yi==1) n01++;
			if(y[i]==1 && yi!=1) n10++;
			if(y[i]==1 && yi==1) n11++;
		}
	}
	if(out==1)
	{
		printf("--------------------------------------------\n");
		printf("              真实正      真实负      准确率\n");
		printf("预测正  %12d%12d%11.2lf%%\n", n11, n01, 100.0*n11/(n11+n01));
		printf("预测负  %12d%12d%11.2lf%%\n", n10, n00, 100.0*n00/(n10+n00));
		printf("召回率  %11.2lf%%%11.2lf%%%11.2lf%%\n", 100.0*n11/(n11+n10), 100.0*n00/(n01+n00),
			100.0*(n00+n11)/num);
		printf("--------------------------------------------\n");
	}
	
	float acc = (float)cor/(float)num;
	printf("* accuracy  = %.2f%% (%d/%d)\n", 100*acc, cor, num);	
	return acc;
}


//-------------------------------------------------------------------------------------------------
// filename, model file
bool NN::LoadModel(const char *filename)
{
	FILE *pfile = fopen(filename, "rb");
	if(pfile==NULL) return false;

	Remove();
	
	// network layers
	fread(&opt, sizeof(opt), 1, pfile);
	fread(&l, sizeof(int), 1, pfile);
	int *nl = new int[l];
	fread(nl, sizeof(int), l, pfile);
	Setup(l, nl);
	delete[] nl;
	
	// parameters of layers
	for(int k=0; k+1<l; k++)
	{
		int nk = n[k], nk1 = n[k+1];
		// W[k] => float[nk1*nk]
		fread(W[k], sizeof(float), nk1*nk, pfile);
		// b[k] => float[nk1]
		fread(b[k], sizeof(float), nk1, pfile);
	}
	
	fclose(pfile);
	return true;
}


//-------------------------------------------------------------------------------------------------
bool NN::SaveModel(const char *filename)
{
	FILE *pfile = fopen(filename, "wb");
	if(pfile==NULL) return false;
	
	// network layers
	fwrite(&opt, sizeof(opt), 1, pfile);
	fwrite(&l, sizeof(int), 1, pfile);
	fwrite(n, sizeof(int), l, pfile);
	
	// parameters of layers
	for(int k=0; k+1<l; k++)
	{
		int nk = n[k], nk1 = n[k+1];
		// W[k] => float[nk1*nk]
		fwrite(W[k], sizeof(float), nk1*nk, pfile);
		// b[k] => float[nk1]
		fwrite(b[k], sizeof(float), nk1, pfile);
	}
	
	fclose(pfile);
	return true;
}
