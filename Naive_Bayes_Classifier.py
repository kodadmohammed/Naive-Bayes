# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:07:51 2019

@author: Achraf
"""
import numpy as np
import collections
import pandas as pn
import math
from statistics import variance

dataset =pn.DataFrame({"person":[0,0,0,0,1,1,1,1],
                       "height":[6,5.92,5.58,5.92,5,5.5,5.42,5.75],
                       "weight":[180,190,170,165,100,150,130,150],
                       "footsize":[12,11,12,10,6,8,7,9]})
test=np.ones(5)
def split(dataset):
    X=dataset.iloc[:,1:] 
    Y=dataset.loc[:,"person"]
    return X,Y


X,Y=split(dataset)


def pdf(x,m,v):
    return 1/math.sqrt(2*math.pi*v)*math.exp(-0.5*math.pow(x-m,2)/v)
    
    

def class_probability(Y):
    Y_dict=collections.Counter(Y)
    n_classes=len(Y_dict)
    prob=np.ones(n_classes)
    nb_elem=Y.shape[0]
    for i in range(n_classes):
        prob[i]=Y_dict[i]/nb_elem
        
    return prob 

classes=class_probability(Y)

def mean_var(dataset):
    Y_dict=collections.Counter(Y)
    n_classes=len(Y_dict)
    nb_elem=Y.shape[0]
    n_cols=dataset.shape[1]-1
    
    m=np.ones((n_classes,n_cols))
    v=np.ones((n_classes,n_cols))
    
    males =dataset.loc[dataset['person']==0,:]
    females =dataset.loc[dataset['person']==1,:]
    for j in range(n_cols):
        col=males.iloc[:,j+1]
        m[0][j]=np.mean(col)
        v[0][j]=variance(col)
        
    for j in range(n_cols):
        col=females.iloc[:,j+1]
        m[1][j]=np.mean(col)
        v[1][j]=variance(col)
    return m,v
        
def probability_features_classes(m,v,sample):
    n_classes=m.shape[0]
    n_cols=m.shape[1]
    probs=np.ones((n_classes,n_cols))
    for i in range(n_classes):
        for j in range(n_cols):
            probs[i][j]=pdf(sample[j],m[i][j],v[i][j])
            
    return probs

def probability_classes(dataset,s):
    Y_dict=collections.Counter(Y)
    n_classes=len(Y_dict)
    n_cols=dataset.shape[1]-1
    p_classes=class_probability(Y)
    m,v=mean_var(dataset)
    probs_cols_cl=probability_features_classes(m,v,s)
    
    probs=np.ones(n_classes)
    for i in range(n_classes):
        produit=1
        for j in range(n_cols):
            produit*=probs_cols_cl[i][j]
        produit*=p_classes[i]
        probs[i]=produit
    return probs

s=[6,130,8]
probs=probability_classes(dataset,s)
