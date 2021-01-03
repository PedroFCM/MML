#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:53:53 2020

@author: elisaaraujo
"""
'Import modules'
import numpy as np
import pandas as pd
# project function database
from function_database import * 
# stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#plots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance



'Import dataset'
#Training
train_labels, train_data = import_dataset('mnist_train.csv')
train_data=np.asarray(train_data) ; train_labels=np.asarray(train_labels)


#Test
test_labels, test_data = import_dataset('mnist_test.csv')
test_data=np.asarray(test_data) ; test_labels=np.asarray(test_labels)

'Get principal directions and train dataset projected'
confidence_level=0.5
vec , train_data_proj , val = pca_projection(train_data,confidence_level)
coef_proj_train = [np.dot(train_data[i], vec) for i in range(len(train_data))]



'Project test_data in principal directions'
test_data=center_matrix(test_data)
test_data_proj=project_dataset(test_data,vec)
coef_proj_test = [np.dot(test_data[i], vec) for i in range(len(test_data))]


distance_list_euc=[] ; distance_list_maha=[]
d_list_euc=[] ; d_list_maha=[]
success_list_euc=[] ; success_list_maha=[]
for j in range (len(test_data_proj[0])):
    for i in range(len(train_data_proj[0])):
        d_list_euc.append(distance.euclidean(train_data_proj[:,i], test_data_proj[:,j]))
        d_list_maha.append(mahalanobis(coef_proj_train[i],coef_proj_test[j],val))
    distance_list_euc.append(min(d_list_euc))
    distance_list_maha.append(min(d_list_maha))
    if train_labels[d_list_euc.index(min(d_list_euc))] == test_labels[j]:
        success_list_euc.append(1)
    else:
        success_list_euc.append(0)
    d_list_euc=[]
    if train_labels[d_list_maha.index(min(d_list_maha))] == test_labels[j]:
        success_list_maha.append(1)
    else:
        success_list_maha.append(0)
    d_list_maha=[]
    if j%100==0:
        print(j)



