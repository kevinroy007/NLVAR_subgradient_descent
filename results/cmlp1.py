
import multiprocessing
from multiprocessing import Lock, Pool
import torch
#torch.set_num_threads(2)
import numpy as np
import matplotlib.pyplot as plt
from synthetic import simulate_var
from models.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized
import pdb
import pickle



#X_np, beta, GC = simulate_var(p=10, T=250, lag=4)

# pickle.dump(X_np,open("VAR_data.txt","wb"))
# pickle.dump(X_np,open("VAR_solid_result/sparse_nonlinear_var_P4N5cbrt_1000_solid_result_l2/VAR_data.txt","wb"))
# pickle.dump(GC,open("results/GC_true_VAR.txt","wb"))
# pickle.dump(GC,open("VAR_solid_result/sparse_nonlinear_var_P4N5cbrt_1000_solid_result_l2/results/GC_true_VAR.txt","wb"))






#X = torch.tensor(X_np[np.newaxis], dtype=torch.float32)


def multiprocess_train_loss_list(lam,T,X,GC,cmlp,l):
 

    

    train_loss_list = train_model_ista(
    cmlp, X, lam, lr=1e-3, lam_ridge=1e-2,  penalty='H', max_iter=5,
    check_every=1)

    # Verify learned Granger causality

    l.acquire()

    GC_est = cmlp.GC().cpu().data.numpy()

    print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
    print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
    print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))

    pickle.dump(GC_est,open("cmlp_results/GC_estimated_cMLP_"+str(lam)+"_.txt","wb"))
    pickle.dump(train_loss_list,open("cmlp_results/train_loss_cMLP.txt_"+str(lam)+"_.txt","wb"))
    
    l.release()


processes = []
lam = np.arange(1, 2, 1)

T = [1000]


pickle.dump(lam,open("cmlp_results/lam_cmlp.txt","ab"))
pickle.dump(T,open("cmlp_results/T.txt","wb"))

pdb.set_trace()





#lock = Lock()

if __name__ ==  '__main__':
    
    l = Lock()
    for i1 in range(len(T)):

        

        X_np = pickle.load(open("A_wAs.txt","rb"))
        GC   = pickle.load(open("A_true_1.txt","rb"))
        X_np = X_np.transpose()
        
        pdb.set_trace()

        X = torch.tensor(X_np[np.newaxis], dtype=torch.float32)
        
        cmlp = cMLP(X.shape[-1], lag=2, hidden=[100])
        
        #pdb.set_trace()


        for i2 in range(len(lam)):

            p = multiprocessing.Process(target = multiprocess_train_loss_list,args = [np.round(lam[i2],4),T[i1],X,GC,cmlp,l])            
            p.start()

            processes.append(p)

        for p1 in processes:
            p1.join()

    






