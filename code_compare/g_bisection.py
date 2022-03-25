import numpy as np
import pdb

def g_bisection(z, i, alpha, w, k, b):
    def sigmoid(x):
        return np.where(x >= 0, 
            1 / (1 + np.exp(-x)), 
            np.exp(x) / (1 + np.exp(x)))

    def f(x,i): 
        N, M = alpha.shape
        a = b[i]
        for m in range(M):
            arg = w[i,m]*x-k[i,m]
            a = a + alpha[i,m] * sigmoid(arg) 
        return a

    max_niter = 1000
    zu = np.sum(alpha[i,:]) + b[i]
    zl = b[i]
    vy = 0   
    if z >= zu or z <= zl:
        pdb.set_trace()
    assert z < zu and z > zl
    yl = -10
    while f(yl,i) > z:
        yl = yl*10
    yu = 10
    while f(yu,i) < z:
        yu = yu*10
    for iter in range(max_niter): 
        vz = f(vy,i) 
        if vz > z:
            yu = vy
        else:
            yl = vy
        #bisection iteration:                         
        vy = (yu + yl) / 2
        if abs(vz-z)<1e-6: #stopping criterion
            break
    return vy