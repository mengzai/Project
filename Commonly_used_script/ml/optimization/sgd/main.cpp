#include <iostream>
#include "data.h"
#include "sgd.h"


//-------------------------------------------------------------------------------------------------
// SGD0, eta=eta0*pow(0.9,epoch)
// lam=0, los=LOG_LOSS,           mis=21.90%
// lam=0, los=HINGE_LOSS,         mis=21.88%
// lam=0, los=SQUARED_HINGE_LOSS, mis=21.91%
// lam=0, los=SMOOTH_HINGE_LOSS,  mis=21.86%
// lam=0, los=SIGMOID_LOSS,       mis=21.99%
//-------------------------------------------------------------------------------------------------
// SGD1, eta=eta0/(1+1.2*t/num)
// lam=0, los=LOG_LOSS,           mis=21.91%
// lam=0, los=HINGE_LOSS,         mis=21.89%
// lam=0, los=SQUARED_HINGE_LOSS, mis=21.92%
// lam=0, los=SMOOTH_HINGE_LOSS,  mis=21.90%
// lam=0, los=SIGMOID_LOSS,       mis=22.01%
//-------------------------------------------------------------------------------------------------
// ASGD, SGD0 + average last epoch
// lam=0, los=LOG_LOSS,           mis=21.86%
// lam=0, los=HINGE_LOSS,         mis=21.84%
// lam=0, los=SQUARED_HINGE_LOSS, mis=21.86%
// lam=0, los=SMOOTH_HINGE_LOSS,  mis=21.83%
// lam=0, los=SIGMOID_LOSS,       mis=21.99%
//-------------------------------------------------------------------------------------------------
int main(int argc, const char* argv[])
{
    const char pathname[] = "/Users/zhengdanian/Downloads/Backup/machine_learning/classification/"
    "stochastic_gradient_descent/sgd/sgd";
    
    char filename0[256], filename1[256];
    sprintf(filename0, "%s/%s", pathname, "alpha_train.bin");
    sprintf(filename1, "%s/%s", pathname, "alpha_test.bin");
    
    Data data;
    if(data.LoadDataBin(true, filename0))
    {
        SGD sgd(SGD::SGD0, 0, SGD::SIGMOID_LOSS, 30);
        sgd.Train(data.n, data.d, data.X, data.yc, NULL);
        
        if(data.LoadDataBin(true, filename1))
        {
            sgd.Test(data.n, data.X, data.yc, NULL, "test: ");
        }
    }
    
    return 0;
}
