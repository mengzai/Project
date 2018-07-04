#include <math.h>
#include <stdio.h>
#include <stack>
#include "cart.h"
#include "heapsort.h"
#pragma warning(disable:4996)


//-------------------------------------------------------------------------------------------------
CART::CART()
{
	criteria = CART_C_GINI;
	max_depth = 5;
	min_samples_leaf = 1;
	d = c = 0;
	cart = 0;
}


//-------------------------------------------------------------------------------------------------
CART::~CART()
{
	Release();
}


//-------------------------------------------------------------------------------------------------
void CART::Release()
{
	if(cart!=0)
	{
		std::stack<treenode*> s;
		s.push(cart);
		
		while(!s.empty())
		{
			treenode *tn = s.top();
			s.pop();
			if(tn->tn0!=0) s.push(tn->tn0);
			if(tn->tn1!=0) s.push(tn->tn1);
			if(tn->mask!=0) delete[] tn->mask;
			delete tn;
		}
	}
}


//-------------------------------------------------------------------------------------------------
void CART::SetParam(CART_CRITERIA criteria, int max_depth, int min_samples_leaf)
{
	this->criteria = criteria;
	this->max_depth = max_depth;
	this->min_samples_leaf = min_samples_leaf;
}


//-------------------------------------------------------------------------------------------------
void CART::SortFeatures()
{
	idn = new int[d*n];

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
}


//-------------------------------------------------------------------------------------------------
void CART::TrainClassifier(int n, int d, float *X, int *y)
{
	// check criteria
	if(criteria>=CART_R_MEAN_SQUARED_ERROR)
	{
		printf("'criteria = %d' is not for classifier\n", criteria);
		return;
	}

	Release();

	this->n = n;
	this->d = d;
	this->X = X;
	this->yc = y;
	this->yr = 0;
	
	SortFeatures();

	// find class number
	c = 0;
	for(int i=0; i<n; i++)
	{
		if(y[i]==-1) y[i] = 0;
		if(c<y[i]) c = y[i];
	}
	c++;
	printf("classifier: num = %d, dim = %d, cls = %d\n", n, d, c);

	// create CART tree
	cart = new treenode();
	memset(cart, 0, sizeof(treenode));
	cart->mask = new bool[n];
	for(int i=0; i<n; i++) cart->mask[i] = true;
	cart->size = n;

	std::stack<treenode*> s;
	s.push(cart);
	while(!s.empty())
	{
		treenode *tn = s.top();
		s.pop();

		if(tn->dept<max_depth)
		{
			float other[4] = {0};
			if(SplitNode(tn->mask, tn->attr, tn->thre, tn->impu, other))
			{
				treenode *tn0 = new treenode();
				treenode *tn1 = new treenode();
				memset(tn0, 0, sizeof(treenode));
				memset(tn1, 0, sizeof(treenode));
				tn0->impu = other[0];
				tn1->impu = other[1];
				tn0->yc = (int)other[2];
				tn1->yc = (int)other[3];
				
				// split the current set into two subsets
				tn0->mask = new bool[n];
				tn1->mask = new bool[n];
				int size0 = 0, size1 = 0;
				float *xattr = X+tn->attr, thre = tn->thre;
				for(int i=0; i<n; i++)
				{
					tn0->mask[i] = tn1->mask[i] = false;
					if(tn->mask[i])
					{
						if(*xattr<=thre) tn0->mask[i] = true, size0++;
						else tn1->mask[i] = true, size1++;
					}
					xattr += d;
				}
				tn0->size = size0;
				tn1->size = size1;

				// add left child and right child
				tn0->dept = tn1->dept = tn->dept+1;
				tn->tn0 = tn0, tn->tn1 = tn1;
				s.push(tn0);
				s.push(tn1);
			}
		}

		// release treenode's mask
		delete[] tn->mask;
		tn->mask = 0;
	}
	delete[] idn;

	// draw cart tree
	DrawCART();
}


//-------------------------------------------------------------------------------------------------
void CART::TrainRegressor(int n, int d, float *X, float *y)
{
	// check criteria
	if(criteria<CART_R_MEAN_SQUARED_ERROR)
	{
		printf("'criteria = %d' is not for regressor\n", criteria);
		return;
	}

	Release();

	this->n = n;
	this->d = d;
	this->X = X;
	this->yc = 0;
	this->yr = y;
	
	SortFeatures();

	// create CART tree
	cart = new treenode();
	memset(cart, 0, sizeof(treenode));
	cart->mask = new bool[n];
	for(int i=0; i<n; i++) cart->mask[i] = true;
	cart->size = n;

	std::stack<treenode*> s;
	s.push(cart);
	while(!s.empty())
	{
		treenode *tn = s.top();
		s.pop();

		if(tn->dept<max_depth)
		{
			float other[4] = {0};
			if(SplitNode(tn->mask, tn->attr, tn->thre, tn->impu, other))
			{
				treenode *tn0 = new treenode();
				treenode *tn1 = new treenode();
				memset(tn0, 0, sizeof(treenode));
				memset(tn1, 0, sizeof(treenode));
				tn0->impu = other[0];
				tn1->impu = other[1];
				tn0->yr   = other[2];
				tn1->yr   = other[3];

				// split the current set into two subsets
				tn0->mask = new bool[n];
				tn1->mask = new bool[n];
				int size0 = 0, size1 = 0;
				float *xattr = X+tn->attr, thre = tn->thre;
				for(int i=0; i<n; i++)
				{
					tn0->mask[i] = tn1->mask[i] = false;
					if(tn->mask[i])
					{
						if(*xattr<=thre) tn0->mask[i] = true, size0++;
						else tn1->mask[i] = true, size1++;
					}
					xattr += d;
				}
				tn0->size = size0;
				tn1->size = size1;

				// add left child and right child
				tn0->dept = tn1->dept = tn->dept+1;
				tn->tn0 = tn0, tn->tn1 = tn1;
				s.push(tn0);
				s.push(tn1);
			}
		}

		// release treenode's mask
		delete[] tn->mask;
		tn->mask = 0;
	}
	delete[] idn;

	// draw cart tree
	DrawCART();
}


//-------------------------------------------------------------------------------------------------
bool CART::SplitNode(bool *mask, int &attr, float &thre, float &impu, float *other)
{
	// check the number of samples in this node
	int m = 0;
	for(int i=0; i<n; i++) if(mask[i]) m++;

	// for root node, need to caculate impurity
	if(m==n)
	{
		if(yc!=0) // classifier
		{
			int *p = new int[c];
			memset(p, 0, c*sizeof(int));
			for(int i=0; i<n; i++) p[yc[i]]++;
			impu = GetNodeImpurity(n, p);
			delete[] p;
		}
		else impu = GetNodeImpurity(n, (int*)yr); // regressor
	}
	
	// if impurity = 0, just return
	if(m<2*min_samples_leaf || impu==0) return false;
	
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

	bool ret = true;

	// classifier
	if(yc!=0)
	{
		// find attr and thre to minimize the impurity
		float impu_min = 1e10;
		int *mc = new int[m*c];
		for(int k=0; k<d; k++) // for each attribute
		{
			int *ik = idm+k*m; // corresponding indexes

			for(int i=0; i<c; i++) mc[i] = 0;
			mc[yc[ik[0]]] = 1;
			for(int i=1; i<m; i++)
			{
				int *mci = mc+i*c;
				for(int j=0; j<c; j++) mci[j] = mci[j-c];
				mci[yc[ik[i]]]++;
			}

			int *mc2 = mc+(m-1)*c;
			//for(int i=0; i<m-1; i++) // for each threshold
			for(int i=min_samples_leaf-1, i1=m-min_samples_leaf; i<i1; i++) // for each threshold
			{
				// threshold must satisfy "x_prev < x_next"
				if(X[ik[i]*d+k]<X[ik[i+1]*d+k])
				{
					// criteria - misclassification
					int mc0 = 0, mc1 = 0, c0 = 0, c1 = 0, *mci = mc+i*c;
					for(int j=0; j<c; j++)
					{
						if(mc0<mci[j]) mc0 = mci[j], c0 = j;
						if(mc1<mc2[j]-mci[j]) mc1 = mc2[j]-mci[j], c1 = j;
					}

					float impu0, impu1, impu01;
					if(criteria==CART_C_MISCLASSIFICATION)
					{
						impu0  = (float)(i+1-mc0)/(float)(i+1);
						impu1  = (float)(m-i-1-mc1)/(float)(m-i-1);
						impu01 = (float)(m-mc0-mc1)/(float)m;
					}
					else
					{
						impu0 = GetNodeImpurity(i+1, mci);
						impu1 = GetNodeImpurity(m-i-1, mc2, mci);
						impu01 = ((i+1)*impu0+(m-i-1)*impu1)/m;
					}
					
					if(impu_min>impu01)
					{
						impu_min = impu01;
						attr = k;
						thre = (X[ik[i]*d+k]+X[ik[i+1]*d+k])/2;
						other[0] = impu0;
						other[1] = impu1;
						other[2] = (float)c0; // label
						other[3] = (float)c1; // label
					}
				}
			}
		}
		delete[] mc;

		//printf("m = %d, attr = %d, thre = %.2f, impu = %.2f -> %.2f\n", m, attr, thre, impu, impu_min);
	}
	
	// regressor
	else
	{
		// find attr and thre to minimize the impurity
		float impu_min = 1e10;
		for(int k=0; k<d; k++) // for each attribute
		{
			int *ik = idm+k*m; // corresponding indexes

			for(int i=0; i<m-1; i++) // for each threshold
			{
				// threshold must satisfy "x_prev < x_next"
				if(X[ik[i]*d+k]<X[ik[i+1]*d+k])
				{
					float mean0 = 0, mean1 = 0;
					float impu0 = GetNodeImpurity(i+1, ik, (int*)&mean0);
					float impu1 = GetNodeImpurity(m-i-1, ik+i+1, (int*)&mean1);
					float impu01 = ((i+1)*impu0+(m-i-1)*impu1)/m;
					
					if(impu_min>impu01)
					{
						impu_min = impu01;
						attr = k;
						thre = (X[ik[i]*d+k]+X[ik[i+1]*d+k])/2;
						other[0] = impu0;
						other[1] = impu1;
						other[2] = mean0;
						other[3] = mean1;
					}
				}
			}
		}

		//printf("m = %d, attr = %d, thre = %.2f, impu = %.2f -> %.2f\n", m, attr, thre, impu,
		//	impu_min);
	}
	//printf("total = %d\n", total);

	delete[] idm;

	return ret;
}


//-------------------------------------------------------------------------------------------------
inline float CART::GetNodeImpurity(int m, int *p)
{
	float impu = 0;
	
	switch(criteria)
	{
	int pmax;
	float *y, ymean;
	case CART_C_GINI:
		for(int i=0; i<c; i++)
		{
			if(p[i]>0)
			{
				float pi = (float)p[i]/(float)m;
				impu += pi*(1-pi);
			}
		}
		break;
	
	case CART_C_CROSS_ENTROPY:
		for(int i=0; i<c; i++)
		{
			if(p[i]>0)
			{
				float pi = (float)p[i]/(float)m;
				impu -= pi*log(pi);
			}
		}
		break;
	
	case CART_C_MISCLASSIFICATION:
		pmax = 0;
		for(int i=0; i<c; i++) if(pmax<p[i]) pmax = p[i];
		impu = (float)(m-pmax)/(float)m;
		break;
	
	case CART_R_MEAN_SQUARED_ERROR:
		y = (float*)p;
		ymean = 0;
		for(int i=0; i<m; i++) ymean += y[i];
		ymean /= m;
		for(int i=0; i<m; i++)
		{
			float temp = y[i]-ymean;
			impu += temp*temp;
		}
		impu /= m;
		break;
	}

	return impu;
}


//-------------------------------------------------------------------------------------------------
inline float CART::GetNodeImpurity(int m, int *p0, int *p1)
{
	float impu = 0;
	
	switch(criteria)
	{
	int pmax;
	float ymean;
	case CART_C_GINI:
		for(int i=0; i<c; i++)
		{
			if(p0[i]-p1[i]>0)
			{
				float pi = (float)(p0[i]-p1[i])/(float)m;
				impu += pi*(1-pi);
			}
		}
		break;
	
	case CART_C_CROSS_ENTROPY:
		for(int i=0; i<c; i++)
		{
			if(p0[i]-p1[i]>0)
			{
				float pi = (float)(p0[i]-p1[i])/(float)m;
				impu -= pi*log(pi);
			}
		}
		break;
	
	case CART_C_MISCLASSIFICATION:
		pmax = 0;
		for(int i=0; i<c; i++) if(pmax<p0[i]-p1[i]) pmax = p0[i]-p1[i];
		impu = (float)(m-pmax)/(float)m;
		break;

	case CART_R_MEAN_SQUARED_ERROR:
		ymean = 0;
		for(int i=0; i<m; i++) ymean += yr[p0[i]];
		ymean /= m;
		for(int i=0; i<m; i++)
		{
			float temp = yr[p0[i]]-ymean;
			impu += temp*temp;
		}
		impu /= m;
		*((float*)p1) = ymean;
		break;
	}

	return impu;
}


//-------------------------------------------------------------------------------------------------
int CART::TestClassifier(int n, float *X, int *y)
{
	if(cart==0){ printf("error: there is no any model\n"); return 0; }
	
	int err = 0, n11 = 0, n10 = 0, n01 = 0, n00 = 0;
	float *xi = X;
	for(int i=0; i<n; i++)
	{
		treenode *tn = cart;
		while(tn->tn0!=0)
		{
			if(xi[tn->attr]<=tn->thre) tn = tn->tn0;
			else tn = tn->tn1;
		}

		int yi = y[i]==-1 ? 0 : y[i];
		if(yi!=tn->yc) err++;
		
		if(c==2)
		{
			if(yi==1 && tn->yc==1) n11++;
			if(yi==1 && tn->yc==0) n10++;
			if(yi==0 && tn->yc==1) n01++;
			if(yi==0 && tn->yc==0) n00++;
		}

		xi += d;
	}

	
	if(c==2)
	{
		printf("--------------------------------------------\n");
		printf("            真实正例    真实负例      准确率\n");
		printf("预测正例%12d%12d%11.2lf%%\n", n11, n01, 100.0*n11/(n11+n01));
		printf("预测负例%12d%12d%11.2lf%%\n", n10, n00, 100.0*n00/(n10+n00));
		printf("召回率  %11.2lf%%%11.2lf%%%11.2lf%%\n", 100.0*n11/(n11+n10), 100.0*n00/(n01+n00), 100.0*(n00+n11)/n);
		printf("--------------------------------------------\n");
	}
	else printf("classification accuracy = %.2f%%\n", 100.0f*(n-err)/n);

	return err;
}


//-------------------------------------------------------------------------------------------------
float CART::TestRegressor(int n, float *X, float *y) // return mean_abs_err
{
	if(cart==0){ printf("error: there is no any model\n"); return 0; }
	
	float err = 0;
	float *xi = X;
	for(int i=0; i<n; i++)
	{
		treenode *tn = cart;
		while(tn->tn0!=0)
		{
			if(xi[tn->attr]<=tn->thre) tn = tn->tn0;
			else tn = tn->tn1;
		}
		err += fabs(tn->yr-y[i]);
		
		xi += d;
	}
	err /= n;
	printf("regressor mean_abs_err = %.2f\n", err);

	return err;
}


//-------------------------------------------------------------------------------------------------
bool CART::SaveModel(const char *filename)
{
	if(cart==0) return false;

	FILE *pfile = fopen(filename, "wb");
	if(pfile==0){ printf("can not open file %s\n", filename); return false; }
	
	// save parameters
	fwrite(&criteria, sizeof(int), 1, pfile); // criteria
	fwrite(&max_depth, sizeof(int), 1, pfile); // max_depth
	fwrite(&min_samples_leaf, sizeof(int), 1, pfile); // max_depth
	fwrite(&d, sizeof(int), 1, pfile); // dimension
	fwrite(&c, sizeof(int), 1, pfile); // class number for classifier

	// save cart tree model
	std::stack<treenode*> s;
	s.push(cart);
	while(!s.empty())
	{
		// save tree node
		treenode *tn = s.top();
		fwrite(&(tn->size), sizeof(int), 1, pfile);
		fwrite(&(tn->dept), sizeof(int), 1, pfile);
		fwrite(&(tn->attr), sizeof(int), 1, pfile);
		fwrite(&(tn->thre), sizeof(float), 1, pfile);
		fwrite(&(tn->impu), sizeof(float), 1, pfile);
		if(criteria<CART_R_MEAN_SQUARED_ERROR) fwrite(&(tn->yc), sizeof(int), 1, pfile);
		else fwrite(&(tn->yr), sizeof(float), 1, pfile);

		// find its two childs
		s.pop();
		if(tn->tn0!=0) s.push(tn->tn0);
		if(tn->tn1!=0) s.push(tn->tn1);
	}
	
	fclose(pfile);
	return true;
}


//-------------------------------------------------------------------------------------------------
bool CART::LoadModel(const char *filename)
{
	FILE *pfile = fopen(filename, "rb");
	if(pfile==0){ printf("can not open file %s\n", filename); return false; }

	// load parameters
	fread(&criteria, sizeof(int), 1, pfile); // criteria
	fread(&max_depth, sizeof(int), 1, pfile); // max_depth
	fread(&min_samples_leaf, sizeof(int), 1, pfile); // max_depth
	fread(&d, sizeof(int), 1, pfile); // dimension
	fread(&c, sizeof(int), 1, pfile); // class number for classifier

	// free old cart tree model
	if(cart!=0) Release();
	cart = new treenode();

	// load cart tree model
	bool ret = true;
	std::stack<treenode*> s;
	s.push(cart);
	while(!s.empty() && ret)
	{
		treenode *tn = s.top();

		// load tree node
		treenode temp;
		memset(&temp, 0, sizeof(treenode));
		if(fread(&(temp.size), sizeof(int), 1, pfile)!=1) break; // last node?
		fread(&(temp.dept), sizeof(int), 1, pfile);
		fread(&(temp.attr), sizeof(int), 1, pfile);
		fread(&(temp.thre), sizeof(float), 1, pfile);
		fread(&(temp.impu), sizeof(float), 1, pfile);
		if(criteria<CART_R_MEAN_SQUARED_ERROR) fread(&(temp.yc), sizeof(int), 1, pfile);
		else fread(&(temp.yr), sizeof(float), 1, pfile);

		if(temp.dept==0)
			memcpy(tn, &temp, sizeof(treenode));
		else
		{
			// find its parent
			while(!s.empty() && temp.dept<=tn->dept)
			{
				s.pop();
				tn = s.top();
			}

			// attach a child
			if(temp.dept==tn->dept+1 && (tn->tn0==0 || tn->tn1==0))
			{
				treenode *temptn = new treenode();
				memcpy(temptn, &temp, sizeof(treenode));
				if(tn->tn1==0) tn->tn1 = temptn;
				else tn->tn0 = temptn;
				s.push(temptn);
			}
			else
			{
				printf("error: one treenode has no parent\n");
				ret = false;
			}
		}
	}
	fclose(pfile);

	if(ret) DrawCART();
	return ret;
}


//-------------------------------------------------------------------------------------------------
void CART::DrawCART()
{
	// draw cart tree
	if(cart!=0)
	{
		std::stack<treenode*> s;
		s.push(cart);
		
		bool leaf = false;
		while(!s.empty())
		{
			treenode *tn = s.top();

			if(leaf){ for(int i=0; i<tn->dept; i++) printf("%20s", ""); }

			char text[256];
			sprintf(text, "%s_%d(%.2f, %d)", tn->tn0==0 ? "leaf" : "node", tn->dept, tn->impu, tn->size);
			printf("%-20s%s", text, tn->tn0==0 ? "\n":"");
			leaf = tn->tn0==0;

			s.pop();
			if(tn->tn0!=0) s.push(tn->tn0);
			if(tn->tn1!=0) s.push(tn->tn1);
		}
	}
}
