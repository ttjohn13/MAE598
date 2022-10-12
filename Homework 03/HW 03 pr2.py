import numpy as np
def f(X1, X2):
    x1 = X1
    x2 = X2
    return 0 -( (4 - 2.1 * x1 **2 + x1 **4 / 3) * x1 **2 + x1 * x2 + (-4 + 4 * x2 **2) * x2 **2)


from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess

cov = squaredExponential()
gp = GaussianProcess(cov, optimize=True, usegrads=False)
acq = Acquisition(mode='ExpectedImprovement')
param = {'X1' : ('cont', [-3, 3]),
         'X2' : ('cont', [-2, 2])}

np.random.seed(20)
gpgo = GPGO(gp,acq,f,param)
gpgo.run(max_iter=15)
gpgo.getResult()
