"""
Created on Sat Jan 11 09:58:56 2020

@author: elisaaraujo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def import_dataset(filename):
    d0 = pd.read_csv(filename)
    labels=d0.loc[:,'label']
    data=d0.iloc[:,1:]
    return labels , data

def plot_digit(label_index,dataset):
    a=np.array(dataset.loc[label_index,:]).reshape(28,28)
    plt.imshow(a)
    plt.show()

def center_matrix(mat):
    mat_avg =np.mean(mat,0) #mean of each column for the entire dataset
    centered_mat = np.asarray(mat-mat_avg) #center the mean in 0
    return centered_mat


def confiança_alvo(p,valprop_cov,traco_cov):
    k=0
    confianca=0
    while confianca < p:
        confianca= confianca+valprop_cov[k]/traco_cov
        k+=1
    return k , confianca

def coef_proj_calc(dcentrado,e_faces,data): 
    for i in range(len(data)):
        coef_proj=[np.dot(dcentrado.loc[i,:], e_faces)]
    return coef_proj

def mahalanobis(coef_proj_train,coef_proj_test,val_train):
    aux=np.multiply((coef_proj_train-coef_proj_test)**2,val_train**(-1))
    return sum(aux)

def pca_projection(data,confidence_level):
    'center the set and normalize the mean to 0'
    sample_data=center_matrix(data)
    'calculate the covariance matrix'
    data_cov = np.matmul(sample_data.T, sample_data)
    'calculate eigeinvalues and eigenvecs'
    valprop_cov, vect_prop_cov = np.linalg.eig(data_cov)
    idx = valprop_cov.argsort()[::-1]
    valprop_cov = np.real(valprop_cov[idx])
    vect_prop_cov = np.real(vect_prop_cov[:,idx])

    'Problem Reduction: get the number of proper values that explain 90% of the variance'
    traco_cov=np.trace(data_cov)
    traco_data=np.trace(data)
    k , confianca = confiança_alvo(confidence_level,valprop_cov,traco_cov)
    print(k)
    val=valprop_cov[:k]
    vec=vect_prop_cov[:,:k]
    sample_data_proj=np.matmul(vec.T,sample_data.T)
    
    return vec , sample_data_proj , val

def project_dataset(data,vec):
    data_proj=np.matmul(vec.T,data.T)
    return data_proj
    
