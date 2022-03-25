import sys

from torch import lu
sys.path.append('code_compare')
import numpy as np
import networkx as nx
from LinearVAR import scaleCoefsUntilStable
from generating import  nonlinear_VAR_realization
import matplotlib.pyplot as plt
from NonlinearVAR import NonlinearVAR
from LinearVAR_Kevin import learn_model as learn_model_linear
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
from learn_model import learn_model
import pdb
import pickle
import os
import csv


p_test = True
os.system("clear")

if p_test:

    

    N=10
    M=10
    T=1000
    P = 2
    NE = 30
    etanl = 0.01 
    etal = 0.001
    lamda = 0.007
    z_data = np.random.rand(N, T)


    # pickle.dump(NE,open("results/NE.txt","wb"))
    # pickle.dump(lamda,open("results/lamda.txt","wb"))

    # A_true =  np.random.rand(N, N, P)



    # for p in range(P): #sparse initialization of A_true

    #      g = erdos_renyi_graph(N, 0.15,seed = None, directed= True)  #remove directed after the meeting 
    #      print(p)
    #      A_t = nx.adjacency_matrix(g)
    #      A_true[:,:,p] = A_t.todense()
    
    # pickle.dump(g,open("results/g.txt","wb")) #note that this connection graph different from A matrix graph 
    # # #because this is the graph for last P.
    
    
    # pickle.dump(A_true,open("results/A_true.txt","wb"))
    
    

    alpha = np.ones((N,M))
    w = np.ones((N,M))
    k = np.ones((N,M))
    b = np.ones((N))
    A = np.ones((N,N,P))

    # pickle.dump(alpha,open("results/alpha.txt","wb"))
    
    # pickle.dump(z_data,open("results/z_data_woAs.txt","wb")) 
   
    # A_true_1 = scaleCoefsUntilStable(A_true, tol = 0.05, b_verbose = False, inPlace=False)

    # pickle.dump(A_true_1,open("results/A_true_1.txt","wb"))

    # z_data =  nonlinear_VAR_realization(A_true_1, T, np.cbrt)
    
    # pdb.set_trace()

    # #following command plots with A matrix scaling
    
    
    z_data = pickle.load(open("results/A_wAs.txt","rb"))

    
    ##########################################################################################

    newobject = NonlinearVAR(N,M,P,filename_tosave = 'model.nlv') #this line meant for comparing the codes if b_comparing = True. However it runs in both cases.


    cost,cost_test,A_n,hat_z_t_n = learn_model(NE, etanl ,z_data, A, alpha, w, k, b,lamda, newobject)

    cost_linear,cost_test_linear,A_l,hat_z_t_l  = learn_model_linear(NE, z_data, A,etal, lamda) 
    
    ##########################################################################################
#    pdb.set_trace()

    pickle.dump(cost,open("results/cost_n_s.txt","wb"))
    pickle.dump(cost_test,open("results/cost_n_test_s.txt","wb"))

    pickle.dump(cost_linear,open("results/cost_linear_s.txt","wb"))
    pickle.dump(cost_test_linear,open("results/cost_linear_test_s.txt","wb"))

    pickle.dump(hat_z_t_n,open("results/hat_z_t_n.txt","wb"))
    pickle.dump(hat_z_t_l,open("results/hat_z_t_l.txt","wb"))

    pickle.dump(A_n,open("results/A_n_s.txt","wb"))
    pickle.dump(A_l,open("results/A_l_s.txt","wb"))

#    pickle.dump(hat_z_t,open("results/hat_z_t.txt","wb"))
    
    
