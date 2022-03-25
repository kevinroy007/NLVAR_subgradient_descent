import sys
from typing import ForwardRef
sys.path.append('NonLinearVAR')
import numpy as np
from compute_gradients import compute_gradients
import cvxpy as cp
import matplotlib.pyplot as plt
from generating import nonlinear_VAR_realization
import pdb
from projection_simplex import projection_simplex_sort as proj_simplex
from LinearVAR import scaleCoefsUntilStable as scaleCoefsUntilStable
from NonlinearVAR import NonlinearVAR

def update_params(eta, z_data, A, alpha, w, k, b, t, z_range):
    N,N,P = A.shape
    N,M = k.shape
    for i in range(N):  
        dC_dA, dc_dalpha_i, dc_dw_i, dc_dk_i, dc_db_i, cost,hat_z_t = compute_gradients( z_data, A, alpha, w, k, b, t, i)

        # projected SGD (stochastic gradient descent (OPTIMIZER)
        alpha[i][:] = alpha[i][:] - eta* dc_dalpha_i
        w[i][:] = w[i][:] - eta* dc_dw_i
        k[i][:] = k[i][:] - eta* dc_dk_i
        # b[i]    = b[i] - eta * dc_db_i TODO
        if np.isnan(alpha).any() or np.isinf(alpha).any():
            print('ERR: found inf or nan in alpha')
            pdb.set_trace()

        #PROJECTION
         
        
        if (alpha[i,:].sum()  !=  z_range[i]): 
            #projection using the code found online
            try:
                alpha[i][:] = proj_simplex(alpha[i][:], z_range[i])
            except Exception:
                print('ERR: exception at proj_simplex')
                pdb.set_trace()
            if abs(np.sum(alpha[i][:])-z_range[i]) > 1e-5:
                print('ERR: projection failed!'); pdb.set_trace()

            #kevins projection will not be used. We can keep the code here for comparison       
            # alpha1 = cp.Variable(M)     
            # cost_i2 = cp.sum_squares(alpha1 - alpha[i,:])
            # obj = cp.Minimize(cost_i2)

            # constr = [sum(alpha1) == z_range[i]]         
            # opt_val = cp.Problem(obj,constr).solve()    
            # alpha_cvxpy =  np.transpose(alpha1.value)

            #if (np.abs(alpha[i][:]-alpha_cvxpy)>1e-5).any():
             #   print('ERR: projections do not coincide!!'); pdb.set_trace()

    
    #pdb.set_trace()
    A  = A - eta*dC_dA

    #pdb.set_trace()
    
    return A, alpha, w, k, b, cost,hat_z_t


def learn_model(NE, eta ,z_data, A, alpha, w, k, b): #TODO: make A, alpha, w, k, b optional
    
    N, T = z_data.shape
    N2,N3,P = A.shape
    assert N==N2 and N==N3
    # document inputs
    
    #TODO randomly initializing A, alpha, w, k, b if not given

    z_maximum  = np.zeros(N)
    for i in range(N):
        z_maximum[i] = np.max(z_data[i,:])
    
    z_minimum  = np.zeros(N)
    for i in range(N):
        z_minimum[i] = np.min(z_data[i,:])
    
    z_range = z_maximum-z_minimum

    z_upper = z_maximum + 0.01*z_range
    z_lower = z_minimum - 0.01*z_range
        
    z_difference = z_upper - z_lower
    b = z_lower
    for i in range(N):
        alpha[i][:] = proj_simplex(alpha[i][:], z_difference[i])
    
    hat_z_t = [0]*int(0.2*T)

    cost_history = np.zeros(NE)
    for epoch in range(NE):  
        cost = np.zeros(T)
        compare_f = np.zeros(T)
        for t in range(P, T):   
            #pdb.set_trace() 
            # hat_z_t = np.zeros(N)    
            # v_z_hat = np.zeros(N) 
            A, alpha, w, k, b, cost[t],hat_z_t[t] = update_params(eta, z_data, A, alpha, w, k, b, t, z_difference)
           
            # pdb.set_trace()
            # v_z_hat = newobject.forward(z_data)
            # compare_f[t] = sum(abs(hat_z_t - v_z_hat))
        
        cost_history[epoch] = sum(cost)
    pdb.set_trace()
    return A, alpha, w, k, b, cost_history
    
    
p_test = False

if p_test:
    N=3
    M=3
    T=10
    P = 3
    NE = 3
    eta = 0.01 
    A_true =  np.random.rand(N, N, P)
    A_true = scaleCoefsUntilStable(A_true, tol = 0.05, b_verbose = False, inPlace=False)
    #print ('A_true is: ', A_true)

    alpha = np.ones((N,M))
    w = np.ones((N,M))
    k = np.ones((N,M))
    b = np.ones((N))
    A = np.ones((N,N,P))

    z_data =  nonlinear_VAR_realization(A_true, T, np.cbrt)

    # plt.plot(z_data[0][:],label = 'sensor 1')
    # plt.plot(z_data[1][:],label = "sensor 2")
    # plt.plot(z_data[2][:],label = "sensor 3")
    # plt.title("VAR with A matrix stabilization")
    # plt.xlabel("Time")
    # plt.ylabel("z_data")
    # plt.legend()
    # plt.show()
    
    newobject = NonlinearVAR(N,M,P,filename_tosave = 'model.nlv')

    A, alpha, w, k, b, cost = learn_model(NE, eta ,z_data, A, alpha, w, k, b)

    #print ('A: ', A)
    #print ('alpha: ', alpha)
    #print ('w: ', w)
    #print ('k: ', k)
    #print ('b: ', b)

    #t1 = np.arange(NE)+1
    #plt.plot(t1,cost)
    #plt.show()
