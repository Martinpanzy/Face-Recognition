#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:34:15 2018

@author: panzengyang
"""

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
import seaborn as sns

'''
# Load image data
mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
face_labels = mat_content['l']

# Split training and testing data under each class
split_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(face_data.T[0:10,:], face_labels.T[0:10,:], test_size=split_ratio)
for i in range(1,52):
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(face_data.T[0+i*10:10+i*10,:], face_labels.T[0+i*10:10+i*10,:], test_size=split_ratio)
    X_train = np.append(X_train, X_train_i, axis=0)
    X_test = np.append(X_test, X_test_i, axis=0)
    y_train = np.append(y_train, y_train_i, axis=0)
    y_test = np.append(y_test, y_test_i, axis=0)
'''
mat_content = sio.loadmat('data.mat')
X_train = mat_content['X_train']
y_train = mat_content['y_train']
X_test = mat_content['X_test']
y_test = mat_content['y_test']
split_ratio = mat_content['split_ratio']

T = 10
class_train_size = int(10*(1-split_ratio))
class_bag_size = 8
for t in range(T):
    index = np.random.choice(class_train_size, class_bag_size)
    X_train_t = X_train[0:8,:][index, :]
    y_train_t = y_train[0:8,:][index, :]
    Mi = np.mean(X_train_t, axis=0)
    Sw = np.zeros((2576,2576))
    for v in X_train_t:
        S0 = np.outer((v - Mi), (v - Mi))
        Sw = Sw + S0
    for i in range(1, 52):
        index = np.random.choice(class_train_size, class_bag_size)
        bagx = X_train[8*i:8*i+8,:][index, :]
        bagy = y_train[8*i:8*i+8,:][index, :]
        mi = np.mean(bagx, axis=0)
        for v in bagx:
            S0 = np.outer((v - mi), (v - mi))
            Sw = Sw + S0
        Mi = np.vstack((Mi, mi))
        X_train_t = np.append(X_train_t, bagx, axis=0)
        y_train_t = np.append(y_train_t, bagy, axis=0)
    if(t == 0):
        Mi_bag = Mi
        Sw_bag = Sw
        X_train_bag = X_train_t
        y_train_bag = y_train_t
    else:
        Mi_bag = np.hstack((Mi_bag, Mi))
        Sw_bag = np.hstack((Sw_bag, Sw))
        X_train_bag = np.hstack((X_train_bag, X_train_t))
        y_train_bag = np.hstack((y_train_bag, y_train_t))

# Determine parameters for feature space randomization
M0 = 50
M1 = 50
T_M1 = 10
Mlda = 27

# Randomisation in feature space
W_OPT = np.empty((T_M1*Mlda,0))
for i in range(T):
    # Extract data for each bag
    bag_idx = range(i*2576,(i+1)*2576)
    X_train = X_train_bag[:,bag_idx]
    Mt = Mi_bag[:,bag_idx]
    Sw = Sw_bag[:,bag_idx]
    # Compute Sb,Sw and global mean
    M = np.mean(Mt, axis=0)
    Sb = np.dot((Mt - M).T,(Mt - M))
    # Compute eigenface for feature space
    A = (X_train - M)
    St = A.dot(A.T)
    e_vals_pca, e_vecs = np.linalg.eig(St)
    e_vecs = np.dot(A.T,e_vecs)
    e_vecs_pca = e_vecs / np.linalg.norm(e_vecs, axis=0)
    # Sorting
    idx_pca=np.argsort(np.absolute(e_vals_pca))[::-1]
    e_vals_pca = e_vals_pca[idx_pca]
    e_vecs_pca = (e_vecs_pca.T[idx_pca]).T
    #randomisation
    Wopt_t = np.empty((0,2576))
    for t in range(T_M1):
        # Randomly select M1 without replacement
        M1_idx = np.random.choice(range(M0,X_train.shape[0]), M1, replace=False)
        Wpca = np.hstack((e_vecs_pca[:,0:M0], e_vecs_pca[:,M1_idx]))
        # Compute eigen space Wlda
        SB = np.dot(np.dot(Wpca.T, Sb), Wpca)
        SW = np.dot(np.dot(Wpca.T, Sw), Wpca)
        e_vals_lda, e_vecs_lda = np.linalg.eig(np.dot(np.linalg.inv(SW), SB))
        # Sort and choose the best Mlda eigenvectors
        idx1 = np.argsort(np.absolute(e_vals_lda))[::-1]
        e_vals_lda = e_vals_lda[idx1]
        e_vecs_lda = (e_vecs_lda.T[idx1]).T
        Wlda = e_vecs_lda[:,:Mlda]
        # Optimal fisherspace
        Wopt_t = np.vstack((Wopt_t, np.dot(Wlda.T, Wpca.T)))     
    W_OPT = np.hstack((W_OPT, Wopt_t))

#'''
#correctness for individual bags
test_size = X_test.shape[0]
correctness = np.zeros((T_M1,T))    
for j in range(T):
    Mt = Mi_bag[:,j*2576:(j+1)*2576]
    M = np.mean(Mt, axis=0)
    X_train = X_train_bag[:,j*2576:(j+1)*2576]
    y_train = y_train_bag[:,j]
    Wopt_t = W_OPT[:,j*2576:(j+1)*2576]
    for k in range(T_M1):
        Wopt = Wopt_t[k*Mlda:(k+1)*Mlda,:]
        mark = 0
        for i in range(test_size):
            W_train = np.dot(X_train - M, Wopt.T)
            W_test = np.dot(X_test[i,:] - M, Wopt.T)
            E = np.linalg.norm(W_test - W_train, axis=1)
            e_idx = np.argmin(E)
            if y_train[e_idx] == y_test[i]:
                mark+=1
        correctness[k,j] = (mark/test_size)*100
#'''


#majority voting
test_size = X_test.shape[0]
mark = 0
for i in range(test_size):   
    for j in range(T):
        bag_idx = range(j*2576,(j+1)*2576)
        X_train = X_train_bag[:,bag_idx]
        y_train = y_train_bag[:,j]
        Mt = Mi_bag[:,bag_idx]
        M = np.mean(Mt, axis=0)
        Wopt_t = W_OPT[:,bag_idx]
        for k in range(T_M1):
            Wopt = Wopt_t[k*Mlda:(k+1)*Mlda,:]
            W_train = np.dot(X_train - M, Wopt.T)
            W_test = np.dot(X_test[i,:] - M, Wopt.T)
            E = np.linalg.norm(W_test - W_train, axis=1)
            e_idx_k = np.argmin(E)
            idx_k = y_train[e_idx_k]
            if(k == 0):
                idx = idx_k
            else:
                idx = np.hstack((idx, idx_k))
        if(j == 0):
            IDX = idx
        else:
            IDX = np.hstack((IDX, idx))
    e = ((T*T_M1 - np.count_nonzero(IDX == y_test[i])) / (T*T_M1))**2
    if(i == 0):
        e_com = e
    else:
        e_com = np.hstack((e_com, e))
    idx_m = np.bincount(IDX).argmax()
    if idx_m == y_test[i]:
        mark+=1
    if(i == 0):
        y_pred = idx_m
    else:
        y_pred = np.hstack((y_pred, idx_m))
E_com = np.mean(e_com, axis = 0)        
crtness = (mark/test_size)*100
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cnf_matrix, cmap="Blues", xticklabels = 5, yticklabels = 5)
plt.show()


'''
#fusion - sum
test_size = X_test.shape[0]
mark = 0
for i in range(test_size):
    for j in range(T):
        bag_idx = range(j*2576,(j+1)*2576)
        X_train = X_train_bag[:,bag_idx]
        y_train = y_train_bag[:,j]
        Mt = Mi_bag[:,bag_idx]
        #M = np.mean(Mt, axis=0)
        Wopt_t = W_OPT[:,bag_idx]
        for k in range(T_M1):
            Wopt = Wopt_t[k*Mlda:(k+1)*Mlda,:]
            for c in range(52):
                M = Mt[c,:]
                wki = np.dot(Wopt,M)
                wkx = np.dot(Wopt,X_test[i,:].T)
                d = np.dot(np.linalg.norm(wkx),np.linalg.norm(wki))
                P = (1+ (np.dot(wkx.T,wki)/d))/2
                if(c == 0):
                    Pk = P
                else:
                    Pk = np.hstack((Pk, P))
            if(k == 0):
                Px = Pk
            else:
                Px = np.vstack((Px, Pk))
        if(j == 0):
            PX = Px
        else:
            PX = np.vstack((PX, Px))
        PX = np.mean(PX, axis = 0)
        y_predsum = np.argmax(PX)
    if(i == 0):
        y_pred_sum = y_predsum
    else:
        y_pred_sum = np.hstack((y_pred_sum, y_predsum))
    if y_predsum == y_test[i]:
        mark+=1
crtness_sum = (mark/test_size)*100 
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred_sum)
sns.heatmap(cnf_matrix, cmap="Blues", xticklabels = 5, yticklabels = 5)
plt.show()
'''