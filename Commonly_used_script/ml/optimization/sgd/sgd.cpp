#include <cmath>
#include <stdio.h>
#include <string.h>
#include "sgd.h"


//-------------------------------------------------------------------------------------------------
SGD::SGD(SGDX sgd, double lam, LOSS los, int epo)
	: sgd(sgd), lam(lam), los(los), epo(epo)
{
    num = 0;
    dim = 0;
    eta0= 0;
    t0 = 0;
    t = 0;
    
    X = NULL;
    y = NULL;
    z = NULL;

    wb= NULL;
    awb = NULL;
}


//-------------------------------------------------------------------------------------------------
SGD::~SGD()
{
    if(wb!=NULL) delete[] wb;
    if(awb!=NULL) delete[] awb;
}


//-------------------------------------------------------------------------------------------------
bool SGD::Train(int num, int dim, float *X, int *y, float *z)
{
    this->num = num;
    this->dim = dim;
    this->lam = lam;
    this->X = X;
    this->y = y;
    this->z = z;
    
    if(wb!=NULL) delete[] wb;
    wb = new double[dim+1];
    memset(wb, 0, (dim+1)*sizeof(double));
    if(sgd==ASGD)
    {
        if(awb!=NULL) delete[] awb;
        awb = new double[dim+1];
        memset(awb, 0, (dim+1)*sizeof(double));
        t0 = (epo-1)*num;
    }
    
    DetermineEta0();
    
    t = 0;
    for(int i=0; i<epo; i++)
    {
        printf("training epoch %d ......\n", i);

        TrainEpoch(i);

        Test(num, X, y, z, "train:");
    }
    
    return true;
}


//-------------------------------------------------------------------------------------------------
void SGD::DetermineEta0()
{
    const double factor = 2.0;
    double loEta = 1;
    double loCost = EvaluateEta(loEta);
    double hiEta = loEta * factor;
    double hiCost = EvaluateEta(hiEta);
    if(loCost < hiCost)
    {
        while (loCost < hiCost)
        {
            hiEta = loEta;
            hiCost = loCost;
            loEta = hiEta / factor;
            loCost = EvaluateEta(loEta);
        }
        eta0 = hiEta;
    }
    else if(hiCost < loCost)
    {
        while(hiCost < loCost)
        {
            loEta = hiEta;
            loCost = hiCost;
            hiEta = loEta * factor;
            hiCost = EvaluateEta(hiEta);
        }
        eta0 = loEta;
    }
    else eta0 = loEta;

    printf("using eta0 = %lf\n", eta0);
}


//-------------------------------------------------------------------------------------------------
double SGD::EvaluateEta(double eta)
{
    eta0 = eta;
    TrainEpoch(0);
    
    char prefix[32];
    sprintf(prefix, "evaluate: eta0 = %7.4lf", eta0);
    double cost = Test(num, X, y, z, prefix);
    
    t = 0;
    memset(wb, 0, (dim+1)*sizeof(double));
    if(sgd==ASGD) memset(awb, 0, (dim+1)*sizeof(double));
    
    return cost;
}


//-------------------------------------------------------------------------------------------------
// perform a training epoch
void SGD::TrainEpoch(int epoch)
{
	double eta = eta0*pow(0.9, epoch); // better

    float *xi = X;
    for(int i=0; i<num; i++)
    {
		// update learning rate eta
        //if(sgd==ASGD) eta = eta0/pow(1+1.2*t/num, 0.75);
        //if(sgd==ASGD) eta = eta0*pow(0.9, t/num);
		//if(sgd==SGD1) eta = eta0/(1+lam*eta0*t); // bad
        if(sgd==SGD1) eta = eta0/(1+1.2*t/num); // good

		// update for regularization term
		if(lam>0)
		{
			double tmp = 1-lam*eta;
			for(int j=0; j<dim; j++) wb[j] *= tmp;
		}

        // perform one iteration of SGD with specified gains
        double s = wb[dim]; // bias
        for(int j=0; j<dim; j++) s += wb[j]*xi[j]; // s += w'*xi
            
        // update for loss term
        double d = Loss(false, s, y[i]); // -dloss/da
        if(d!=0)
        {
            double tmp = eta*d;
            for(int j=0; j<dim; j++) wb[j] += tmp*xi[j];
            
            // same for bias
            wb[dim] += 0.01*tmp;
        }

        // averaging
        if(sgd==ASGD && t>=t0)
        {
            // a := a + mu_t (w-a)
            double mu = 1/(1+t-t0);
            for(int j=0; j<=dim; j++) awb[j] += mu*(wb[j]-awb[j]);
        }
            
        xi += dim;
        t++;
    }
}


//-------------------------------------------------------------------------------------------------
// perform a test pass
double SGD::Test(int num, float *X, int *y, float *z, const char *prefix)
{
    double *wbt = (t<=t0 || awb==NULL) ? wb : awb; // SGD or ASGD

	double nerr = 0;
    double loss = 0;
    float *xi = X;
    for(int i=0; i<num; i++)
    {
        // compute the output for one example
        double s = wbt[dim];
        for(int j=0; j<dim; j++) s += wbt[j]*xi[j];

        loss += Loss(true, s, y[i]);
        if(s*y[i]<=0) nerr++;
        
        xi += dim;
    }
    nerr /= num;
    loss /= num;
    
    // compute the cost
    double wnorm = 0;
	if(lam>0){ for(int i=0; i<dim; i++) wnorm += wbt[i]*wbt[i]; }
    double cost = loss + 0.5*lam*wnorm;
    
    printf("%s loss = %.4lf, cost = %.4lf, misclassification = %.2lf%%\n", prefix, loss, cost,
           100*nerr);
    return cost;
}


//-------------------------------------------------------------------------------------------------
// b - true  return loss(a,y)
//     false return -dloss(a,y)/da
double SGD::Loss(bool b, double a, double y)
{
    double z = a*y;
    switch(los)
    {
    // logloss(a,y) = log(1+exp(-a*y))
    case LOG_LOSS:
        if(b)
        {
            if(z> 18) return exp(-z);
            if(z<-18) return -z;
            return log(1+exp(-z));
        }
        else
        {
            if(z> 18) return y*exp(-z);
            if(z<-18) return y;
            return y/(1+exp(z));
        }
    
    // hingeloss(a,y) = max(0, 1-a*y)
    case HINGE_LOSS:
        if(b)
        {
            if(z>1) return 0;
            return 1-z;
        }
        else
        {
            if(z>1) return 0;
            return y;
        }
    
    // squaredhingeloss(a,y) = 1/2 * max(0, 1-a*y)^2
    case SQUARED_HINGE_LOSS:
        if(b)
        {
            if(z>1) return 0;
            double d = 1-z;
            return 0.5*d*d;
        }
        else
        {
            if(z>1) return 0;
            return y*(1-z);
        }
    
    // smoothhingeloss(a,y) = if z>1, 0; if 0<=z<=1, 0.5*(1-z)^2; if z<0, 0.5-z.
    case SMOOTH_HINGE_LOSS:
        if(b)
        {
            if(z>1) return 0;
            if(z<0) return 0.5-z;
            double d = 1-z;
            return 0.5*d*d;
        }
        else
        {
            if(z>1) return 0;
            if(z<0) return y;
            return y*(1-z);
        }
    
    // sigmoid(a,y) = 1/(1+exp(a*y))
    case SIGMOID_LOSS:
        if(b)
        {
            if(z<0) return 1/(1+exp(z));
            double d = exp(-z);
            return d/(1+d);
        }
        else
        {
            double d = z>0 ? exp(-z) : exp(z);
            return y*d/((1+d)*(1+d));
        }
    }
    return 0;
}
