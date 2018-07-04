#ifndef sgd_h
#define sgd_h

//-------------------------------------------------------------------------------------------------
// Stochastic Gradient Descent (SGD)
// Averaged Stochastic Gradient Descent (ASGD)
//-------------------------------------------------------------------------------------------------
// Obj(w) = 1/2 lambda w^2  + 1/n sum_i=1^n L(x_i,w)
// SGD0: w := (1 - lambda eta_t) w - eta_t dL/dw(x,w)
//       where eta = eta0 * 0.9 ^ epoch
// SGD1: w := (1 - lambda eta_t) w - eta_t dL/dw(x,w)
//       where eta_t = eta0/(1+1.2*t/num)
// ASGD: w := (1 - lambda eta_t) w - eta_t dL/dw(x,w)
//       where eta = eta0 * 0.9 ^ epoch
//       a := a + (w-a)/(1+t-t0), t>=t0
//-------------------------------------------------------------------------------------------------
class SGD
{
public:
    enum SGDX { SGD0, SGD1, ASGD };
    enum LOSS { LOG_LOSS, HINGE_LOSS, SQUARED_HINGE_LOSS, SMOOTH_HINGE_LOSS, SIGMOID_LOSS };
    SGD(SGDX sgd, double lam, LOSS los, int epo);
    ~SGD();
    
    bool Train(int num, int dim, float *X, int *y, float *z);

    // perform a test pass
    double Test(int num, float *X, int *y, float *z, const char *prefix);

private:
    void DetermineEta0();
    double EvaluateEta(double eta);

    // perform a train pass
    void TrainEpoch(int epoch);
    
    // b - true  return loss(a,y)
    //     false return -dloss(a,y)/da
    double Loss(bool b, double a, double y);
    
    // training
    SGDX   sgd; // SGD0, SGD1, ASGD1
    int    num; // training sample number
    int    dim; // feature dimension
    double lam; // regularization parameter
    LOSS   los; // loss function
    int    epo; // training epochs
    double eta0;// initial learning rate eta0
    double t0;  //
    double t;   // iterations

    float  *X;  // pointer to training samples
    int    *y;  // pointer to labels
    float  *z;  // pointer to training sample weights
    
    // model
    double *wb; // linear model [w b]
    double *awb;// linear model [aw ab]
};

#endif
