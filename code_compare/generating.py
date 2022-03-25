import numpy as np

def linear_VAR_realization(A_true,T):
    N, N2, P = A_true.shape
    z_data = np.zeros((N,T))
    z_data[:,0:P] = np.random.rand(N, P)
    for t in range (P, T):
        for p in range(1,P+1):
            epsilon_t = np.random.randn((N)) * 0.01
            z_data[:, t] =  z_data[:, t] +  A_true[:,:, p-1] @ z_data[:, t-p] + epsilon_t
            # print z_data for each sensor to check the stability and compare with luismis code for doubht for generation part
    return z_data


# if the server output is not good, try generationg synthetica data witht his funciton

# def VAR_realization(A_true,T): 
#     N,N1,P = A_true.shape; assert(N == N1)
#     u = np.random.rand(N, T)*0.01
#     y= np.zeros((N,T)) 
#     y[:, 0:P] = np.random.rand(N, P)
#     for t in range (P, T):
#         for i in range(N):
#             y[i,t] = 0 
#             for k in range(N):
#                 for p in range(1,P+1):
#                     y[i,t] = y[i,t] + A_true[i,k,p-1]*y[k,t-p] + u[i,t]
#     return y

def nonlinear_VAR_realization(A_true, T, nonlinearity):
    y_data = linear_VAR_realization(A_true,T)

    z_data = nonlinearity(y_data)
    return z_data

def cube(x):
    return x**3

# unit test

b_test = False

if b_test:
    
    A_true = np.ones((2, 2, 4))
    T = 10
    my_nonlinearity = cube

    my_y = linear_VAR_realization(A_true,T)
    my_z = nonlinear_VAR_realization(A_true,T, my_nonlinearity)

    print(my_y)
    print(my_z)