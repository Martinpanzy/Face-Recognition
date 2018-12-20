#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:21:42 2018

@author: panzengyang
"""

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np
import scipy.io as sio
import pr_functions as pr
import matplotlib.pyplot as plt
import seaborn as sns


#Load image data
mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
face_labels = mat_content['l']

# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(face_data.T[0:10,:], face_labels.T[0:10,:], test_size=0.2)
Mi = np.mean(X_train, axis=0)
Sw = np.zeros((2576,2576))
for v in X_train:
    S0 = np.outer((v - Mi), (v - Mi))
    Sw = Sw + S0

for i in range(1, 52):
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(face_data.T[0+i*10:10+i*10,:], face_labels.T[0+i*10:10+i*10,:], test_size=0.2)
    mi = np.mean(X_train_rand, axis=0)
    for v in X_train_rand:
        S0 = np.outer((v - mi), (v - mi))
        Sw = Sw + S0
    Mi = np.vstack((Mi, mi))
    X_train = np.append(X_train, X_train_rand, axis=0)
    X_test = np.append(X_test, X_test_rand, axis=0)
    y_train = np.append(y_train, y_train_rand, axis=0)
    y_test = np.append(y_test, y_test_rand, axis=0)

# Compute Sb and mean face
M = np.mean(Mi, axis=0)
Sb = np.dot((Mi - M).T,(Mi - M))

# Compute the rank of Sb and Sw
rank_Sb = np.linalg.matrix_rank(Sb)
rank_Sw = np.linalg.matrix_rank(Sw)

# Compute eigenface for feature space
A = (X_train - M)
St = A.dot(A.T)
e_vals_pca, e_vecs = np.linalg.eig(St)
e_vecs = np.dot(A.T,e_vecs)
e_vecs_pca = e_vecs / np.linalg.norm(e_vecs, axis=0)

# Sort and pick the best Mpca eigenvectors
idx_pca=np.argsort(np.absolute(e_vals_pca))[::-1]
e_vals_pca = e_vals_pca[idx_pca]
e_vecs_pca = (e_vecs_pca.T[idx_pca]).T
rank_pca = np.linalg.matrix_rank(e_vecs_pca)
Mpca = 100
Wpca = e_vecs_pca[:,:Mpca]

# Compute eigen space Wlda
SB = np.dot(np.dot(Wpca.T, Sb), Wpca)
SW = np.dot(np.dot(Wpca.T, Sw), Wpca)
e_vals_lda, e_vecs_lda = np.linalg.eig(np.dot(np.linalg.inv(SW), SB))

# Sort and choose the best Mlda eigenvectors
idx1=np.argsort(np.absolute(e_vals_lda))[::-1]
e_vals_lda = e_vals_lda[idx1]
e_vecs_lda = (e_vecs_lda.T[idx1]).T
Mlda = 27
Wlda = e_vecs_lda[:,:Mlda]

# Optimal fisherspace
Wopt_t = np.dot(Wlda.T, Wpca.T)

'''
#pr.plot_graph("bar", Wopt_t, Mlda, 'index', 'eigen_value', 1, 100000, 'Outputs_c/eigenvlues')
# Plot the best Mlda fisherfaces
for i in range(Mlda):
    pr.plot_image(Wopt_t.T[:,i], 46, 56, 'Outputs_c/fisher_faces_'+str(i))

# Pick test face for projection and reconstruction
test = 15
W_test = np.dot(X_test[test,:] - M, Wopt_t.T)
Xa=Wopt_t[:Mlda,:].T*W_test
X_reconst=M+np.sum(Xa, axis=1)

# Print the original face and reconstrcuted face for comparison
pr.plot_image(X_test[test,:], 46, 56, 'Outputs_c/test_face_M='+str(Mlda))
pr.plot_image(X_reconst, 46, 56, 'Outputs_c/reconstructed_face_M='+str(Mlda))


# Classification using NN
test_size = X_test.shape[0]
correctness = np.zeros(Mlda)
for m in range(Mlda):
    mark = 0
    Wlda = e_vecs_lda[:,:m]
    Wopt_t = np.dot(Wlda.T, Wpca.T)
    for i in range(test_size):
       W_train = np.dot(X_train - M, Wopt_t.T)
       W_test = np.dot(X_test[i,:] - M, Wopt_t.T)
       E = np.linalg.norm(W_test - W_train, axis=1)
       e_idx = np.argmin(E)
       if y_train[e_idx] == y_test[i]:
           mark+=1
    correctness[m] = (mark/test_size)*100
    
pr.plot_graph("line", correctness, Mlda, 'Mlda', 'success rate', 5, 5, 'Outputs_c/success_rate_against_M')
print("Highest succes rate %.2f%% when M = %d" % (np.max(correctness), np.argmax(correctness)))
'''



#confusion matrix
grade = 0
test_size = X_test.shape[0]
for i in range(test_size):
    W_train = np.dot(X_train - M, Wopt_t.T)
    W_test = np.dot(X_test[i,:] - M, Wopt_t.T)
    E = np.linalg.norm(W_test - W_train, axis=1)
    e_idx = np.argmin(E)
    idx = y_train[e_idx]
    if(i == 0):
        E_idx = e_idx
        y_pred = idx
    else:
        E_idx = np.hstack((E_idx, e_idx))
        y_pred = np.hstack((y_pred, idx))
    if(idx == y_test[i]):
        grade = grade + 1
cnf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(cnf_matrix, cmap="Blues", xticklabels = 5, yticklabels = 5)
plt.show()
grade = grade/test_size




'''
#heatmap 
Mpca = 50
Mlda = 25
recog_acc_mt = np.zeros((Mlda, Mpca))
test_size = X_test.shape[0]
for p in range(Mpca):
    Wpca = e_vecs_pca[:,:(p*8)]
    for l in range(Mlda):
        mark = 0
        # Compute eigen space Wlda
        SB = np.dot(np.dot(Wpca.T, Sb), Wpca)
        SW = np.dot(np.dot(Wpca.T, Sw), Wpca)
        e_vals_lda, e_vecs_lda = np.linalg.eig(np.dot(np.linalg.inv(SW), SB))
        # Sort and choose the best Mlda eigenvectors
        idx1=np.argsort(np.absolute(e_vals_lda))[::-1]
        e_vals_lda = e_vals_lda[idx1]
        e_vecs_lda = (e_vecs_lda.T[idx1]).T
        Wlda = e_vecs_lda[:,:(l*2)]
        # Optimal fisherspace
        Wopt_t = np.dot(Wlda.T, Wpca.T)
        for i in range(test_size):
           W_train = np.dot(X_train - M, Wopt_t.T)
           W_test = np.dot(X_test[i,:] - M, Wopt_t.T)
           E = np.linalg.norm(W_test - W_train, axis=1)
           e_idx = np.argmin(E)
           if y_train[e_idx] == y_test[i]:
               mark+=1
        recog_acc_mt[l,p] = mark/test_size

ax = sns.heatmap(recog_acc_mt, cmap="Blues", robust = True, square = True, xticklabels = 'auto', yticklabels = 'auto')

ax.axhline(y=0, color='k',linewidth=3)
ax.axvline(x=0, color='k',linewidth=3)
ax.axhline(y=Mlda, color='k',linewidth=3)
ax.axvline(x=Mpca, color='k',linewidth=3)
plt.show()
'''