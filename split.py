#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:22:38 2018

@author: panzengyang
"""

from __future__ import print_function
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.io as sio

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