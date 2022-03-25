import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pdb
from sklearn import metrics

A_true = pickle.load(open("A_true.txt","rb"))
A_true_1 = pickle.load(open("A_true_1.txt","rb"))
z_data = pickle.load(open("z_data_woAs.txt","rb")) 
z_data_1 = pickle.load(open("A_wAs.txt","rb"))
cost_n = pickle.load(open("cost_n.txt","rb"))
cost_n_test = pickle.load(open("cost_n_test.txt","rb"))
cost_linear = pickle.load(open("cost_linear.txt","rb"))
cost_linear_test = pickle.load(open("cost_linear_test.txt","rb"))
A_n = pickle.load(open("A_n.txt","rb"))
A_l = pickle.load(open("A_l.txt","rb"))
alpha = pickle.load(open("alpha.txt","rb"))
g = pickle.load(open("g.txt","rb"))
lamda = pickle.load(open("lamda.txt","rb"))
NE = pickle.load(open("NE.txt","rb"))
hat_z_t = pickle.load(open("hat_z_t.txt","rb"))


lam = NE = pickle.load(open("cmlp_results/lam_cmlp.txt","rb"))
GC_tank = pickle.load(open("cmlp_results/GC_estimated_cMLP_"+str(lam[0])+"_.txt","rb"))


print(lamda)

N,N,P = A_true.shape
N,T  = z_data.shape
N,M = alpha.shape

figure, axis = plt.subplots(2, 2)

figure.suptitle(" N = "+str(N)+" P = "+str(P)+" T = "+str(T)+ " lambda = "+str(lamda)+" M = "+str(M))

#following command plots without A matrix scaling

axis[0, 0].plot(z_data[0][:],label = 'sensor 1')
axis[0, 0].plot(z_data[1][:],label = "sensor 2")
axis[0, 0].plot(z_data[2][:],label = "sensor 3")
axis[0, 0].set_title("VAR without A matrix stabilization")
axis[0, 0].set_xlabel("Time")
axis[0, 0].set_ylabel("z_data")
axis[0, 0].legend()

axis[0, 1].plot(z_data_1[0][:],label = 'sensor 1')
axis[0, 1].plot(z_data_1[1][:],label = "sensor 2")
axis[0, 1].plot(z_data_1[2][:],label = "sensor 3")
axis[0, 1].set_title("VAR with A matrix stabilization")
axis[0, 1].set_xlabel("Time")
axis[0, 1].legend()

t1 = np.arange(NE)+1
axis[1, 0].plot(t1,cost_n,label = "NonLinear VAR_train")
axis[1, 0].plot(t1,cost_n_test,label = "NonLinear VAR_test")

axis[1, 0].plot(t1,cost_linear,label = "Linear VAR_train")
#axis[1, 0].plot(t1,cost_linear_test,label = "Linear VAR_test")
axis[1, 0].set_title("Cost cmparison LinearVAR vs Non LinearVAR")
axis[1, 0].set_xlabel("Epoch")
axis[1, 0].set_ylabel("cost")
axis[1, 0].legend()


t1 = np.arange(NE)+1
axis[1, 1].plot(t1,cost_n,label = "NonLinear VAR")
axis[1, 1].set_title("Cost magnification of Non LinearVAR")
axis[1, 1].set_xlabel("Epoch")
axis[1, 1].legend()


fig = plt.figure()
fig.suptitle(" N = "+str(N)+" P = "+str(P)+" T = "+str(T)+ " lambda = "+str(lamda)+" M = "+str(M))
ax1 = fig.add_subplot(3,4,1)
ax1a = fig.add_subplot(3,4,2)
ax1b = fig.add_subplot(3,4,3)
ax1c = fig.add_subplot(3,4,4)

ax2 = fig.add_subplot(3,4,5)
ax2a = fig.add_subplot(3,4,6)
ax2b = fig.add_subplot(3,4,7)
ax2c = fig.add_subplot(3,4,8)

ax3 = fig.add_subplot(3,4,9)
ax3a = fig.add_subplot(3,4,10)
ax3b = fig.add_subplot(3,4,11)
ax3c = fig.add_subplot(3,4,12)

ax1.set_ylabel('Ture Adjacency matrix', fontsize=16)
ax2.set_ylabel('Linear VAR', fontsize=16)
ax3.set_ylabel(' Non linear VAR', fontsize=16)

ax1.title.set_text('P = 1')
ax1a.title.set_text('P = 2')
ax1b.title.set_text('P = 3 ')
ax1c.title.set_text('P = 4')



cb1 =  ax1.imshow(A_true_1[:,:,0], vmin=0, vmax=1, cmap='jet', aspect='auto')
cb1a =  ax1a.imshow(A_true_1[:,:,1], vmin=0, vmax=1, cmap='jet', aspect='auto')


cb2 =  ax2.imshow(A_l[:,:,0], vmin=0, vmax=1, cmap='jet', aspect='auto')
cb2a =  ax2a.imshow(A_l[:,:,1], vmin=0, vmax=1, cmap='jet', aspect='auto')


cb3 =  ax3.imshow(A_n[:,:,0], vmin=0, vmax=1, cmap='jet', aspect='auto')
cb3a =  ax3a.imshow(A_n[:,:,1], vmin=0, vmax=1, cmap='jet', aspect='auto')


fig.colorbar(cb1,ax = ax1c,orientation='vertical')
fig.colorbar(cb2,ax = ax2c,orientation='vertical')
fig.colorbar(cb3,ax = ax3c, orientation='vertical')


#remove after meeting
# fig = plt.figure()
# fig.suptitle(" N = "+str(N)+" P = "+str(P)+" T = "+str(T)+ " lambda = "+str(lamda)+" M = "+str(M))

#nx.draw(g)

def roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []

    for threshold in thresholds:
         
        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return fpr, tpr


fprP_n = []*0
tprP_n = []*0
fprP_l = []*0
tprP_l = []*0

thresholds = np.arange(-1,0.4,0.01)

for p in range(P):
    fpr = []*0
    tpr = []*0

    fpr,tpr = roc_curve(A_true[:,:,p],A_n[:,:,p],thresholds)

    fprP_n.append(fpr)
    tprP_n.append(tpr)

for p in range(P):
    fpr = []*0
    tpr = []*0

    fpr,tpr= roc_curve(A_true[:,:,p],A_l[:,:,p],thresholds)

    fprP_l.append(fpr)
    tprP_l.append(tpr)

#pdb.set_trace()

figure, axis = plt.subplots(1, P)
AUC_n = []
AUC_l = []
for p in range(P):
    
    
    axis[ p].plot(fprP_n[p],tprP_n[p],'-bo', label='Nonlinear_VAR ')
    axis[ p].plot(fprP_l[p],tprP_l[p],'-ro', label='linear_VAR ')
    axis[ p].set_ylabel("tpr")
    axis[ p].set_xlabel("fpr")
    axis[ p].set_title("ROC__ "+str(P))
    # for i2 in range(len(thresholds)):
    #     axis[ p].text(fprP_n[p][i2],tprP_n[p][i2],str(np.round(thresholds[i2],5)))
    axis[ p].legend()

    #axis[ p].legend()
    AUC_n.append(metrics.auc(fprP_n[p], tprP_n[p]))
    AUC_l.append(metrics.auc(fprP_l[p], tprP_l[p]))
# figure.suptitle("Hyperparmeter sweep")

print(AUC_n)
print(AUC_l)

T1 = np.arange(int(T*0.8),T,1)
figure, axi = plt.subplots()

axi.plot(T1,hat_z_t[1,int(T*0.8):T],'-bo', label='linear_VAR ')
axi.plot(T1,z_data_1[1,int(T*0.8):T],'-ro', label='true_signal ')
axi.set_ylabel("sensor measurement")
axi.set_xlabel("time stamps")
axi.set_title("ROC__ "+str(P))
axi.legend()

plt.show()
