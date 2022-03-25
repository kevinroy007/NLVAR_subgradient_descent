import torch
#torch.set_num_threads(2)

import numpy as np
import matplotlib.pyplot as plt
#from cmlp1 import X_np
from synthetic import simulate_lorenz_96
from models.crnn import cRNN, train_model_ista
import pickle
import pdb
import multiprocessing
from multiprocessing import process, Lock
#T = 250
lag = 4

#X_np, GC = simulate_lorenz_96(p=10, F=10, T=1000)



def multiprocess_train_loss_list(lam,T,X,GC,crnn):
    
    

    train_loss_list = train_model_ista(
        crnn, X, lam, lam_ridge=1e-2, lr=5e-2, max_iter=20000,context=10,
        check_every=1)

    GC_est = crnn.GC().cpu().data.numpy()


    print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
    print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
    print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))
    
    pickle.dump(GC_est,open("crnn_results/results_"+str(T)+"/GC_estimated_cRNN_"+str(lam)+"_.txt","wb"))
    pickle.dump(train_loss_list,open("crnn_results/results_"+str(T)+"/train_loss_cRNN.txt_"+str(lam)+"_.txt","wb"))
    
    


processes = []

lam = np.arange(0.001, 0.01, 0.001)
T = [250,500,1000]

pickle.dump(lam,open("crnn_results/lam_cRNN.txt","wb"))
pickle.dump(T,open("crnn_results/T.txt","wb"))

pdb.set_trace()

if __name__ ==  '__main__':


    for i1 in range(len(T)):
        
        X_np = pickle.load(open("VAR_data_"+str(T[i1])+"_.txt","rb"))
        GC = pickle.load(open("GC_true_VAR_"+str(T[i1])+"_.txt","rb"))

        X = torch.tensor(X_np[np.newaxis], dtype=torch.float32)

        crnn = cRNN(X.shape[-1], hidden=100)
        

        #pdb.set_trace()

        for i2 in range(len(lam)):

            p = multiprocessing.Process(target = multiprocess_train_loss_list,args = [np.round(lam[i2],4),T[i1],X,GC,crnn])            
            p.start()

            processes.append(p)

        for p1 in processes:
            p1.join()
