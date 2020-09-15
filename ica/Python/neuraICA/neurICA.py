#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:16:34 2020

@author: lyp
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import Input
from keras.models import Model
from keras import backend as K
from keras import regularizers
import tensorflow as  tf

import scipy.io as scio

def gauss_standard(x):
    return np.exp(-0.5*np.square(x))/(np.sqrt(2*np.pi))

def scale(X,center=True,scale=True):
    if(center==True):
        X =X -np.mean(X,axis=0)
    if(scale==True):
        X =X/np.std(X,axis=0)
    return X
    
def whiten(X):
    [N,p] =X.shape
    cov =(X.T@X)/N
    [eig_vals,eig_vecs] =np.linalg.eig(cov)
    P =(np.diag(1.0/np.sqrt(eig_vals)))@eig_vecs.T
    return [P,(P@X.T).T]

def Amari_metric(A0,A):
    R =np.abs(A0@np.linalg.inv(A))
    m =A0.shape[0]
    row_max =np.max(R,axis=1)
    col_max =np.max(R,axis=0)
    return -1.0+(np.sum(np.sum(R,axis=1)/row_max)+np.sum(np.sum(R,axis=0)/col_max))/(2*m)
    
def mixmat(m):
    A =np.random.normal(size=(m,m))
    [u,s,vh] =np.linalg.svd(A,full_matrices=True)
    d =np.sort(np.random.uniform(low=0.0,high=1.0,size=(m,)))+1.0
    print('condition ',d[-1]/d[0])
    A =u@vh.T@np.diag(d)
    return A

def orth(W):
    [u,s,vh] =np.linalg.svd(W,full_matrices=True)
    W =u@vh
    return W


def mdi_loss(y_true,y_pred):
    #怎么不是在最小化，而是在增大
    norm_val =K.exp(y_pred)*y_true
    #return (K.log(K.mean(norm_val,axis=-1))-(K.mean(norm_val*y_pred,axis=-1)/K.mean(norm_val,axis=-1)))
    return ((K.log(K.mean(norm_val,axis=-1))-(K.mean(y_pred,axis=-1)/(K.mean(norm_val,axis=-1)+K.epsilon()))))
def mdi_loss2(y_true,y_pred):
    norm_val =K.exp(y_pred)*y_true
    return K.mean(norm_val,axis=-1)-K.mean(y_pred,axis=-1)

def mdi_loss3(y_true,y_pred):
    norm_val =K.exp(y_pred)*y_true
    return K.mean(norm_val,axis=-1)*K.log(K.mean(norm_val,axis=-1))-K.mean(y_pred,axis=-1)

def negentropy(y_true,y_pred):
    return K.mean(y_pred,axis=-1)


def neuralICA(X,W0,M=1024,maxiters=20,epochs=400,optimizer='rmsprop',tol=1e-7,activation='sigmoid',lbd=0.05):
    [N,p] =X.shape
    batch_size =N
    m =p
    models =[]
    inputs =[]
    outputs =[]
    loss     =np.zeros((maxiters,m),dtype=float)
    metrices =np.zeros((maxiters,m),dtype=float)
    y_pred =np.zeros((N,m),dtype=float)
    tol =1e-7
    W=W0
    W_last =np.ones(W0.shape,dtype=float)
    for i in range(m):
        input_tensor =Input(shape=(1,),dtype='float32')
        x =layers.Dense(M,activation=activation,kernel_initializer='random_uniform',
                bias_initializer='zeros')(input_tensor)
        #x =layers.Dense(50,activation='sigmoid',kernel_initializer='random_uniform',
        #            bias_initializer='zeros',kernel_regularizer=regularizers.l1(1))(x)
        output_tensor =layers.Dense(1,activation="linear",kernel_initializer='random_uniform',
                bias_initializer='zeros',kernel_regularizer=regularizers.l2(lbd))(x)
        model =Model(input_tensor,output_tensor)
        model.compile(optimizer=optimizer,loss=mdi_loss2,metrics=[negentropy])
        models.append(model)
        inputs.append(input_tensor)
        outputs.append(output_tensor)

    for i in range(maxiters):
        X_train =(W@X.T).T
        y_train =gauss_standard(X_train)
        if(Amari_metric(W_last,W)<tol):
            break
        W_last =W.copy()
        for j in range(m):
            for k in range(epochs):
                #print('k ',k)
                [loss[i,j],metrices[i,j]] =models[j].train_on_batch(x=X_train[:,j],y=y_train[:,j])
                grad  =K.gradients(models[j].output,[models[j].input])[0]
                grad2 =K.gradients(grad,[models[j].input])[0]
                grads =K.function([models[j].input],[grad,grad2])
                [grad_val,grad2_val] =grads([X_train[:,j][:,np.newaxis]])
                W[j,:] =np.mean(X*grad_val,axis=0)-np.mean(grad2_val)*W[j,:]
        W =orth(W)
        
        print('iteration ',i,' loss ',loss[i,:],' negentropy ',metrices[i,:])
    return [W,metrices]
        




path =r"./data/dists.mat"
data = scio.loadmat(path)["dists"]


m  =2
S  =data[[15,5],[3,4],:].T
A0 =mixmat(2)
X  =(A0@S.T).T
X  =scale(X,center=True,scale=False)
[P,X] =whiten(X)
target =np.linalg.inv(P@A0)
W0 =np.random.normal(size=(2,2))
W0 =orth(W0)

[W,metrics]=neuralICA(X,W0=W0,M=1024,maxiters=20,epochs=30,optimizer='rmsprop',tol=1e-7,activation='sigmoid',lbd=0.02)
