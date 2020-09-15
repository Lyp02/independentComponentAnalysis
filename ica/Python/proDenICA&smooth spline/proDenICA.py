#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 05:10:01 2020

@author: lyp
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from smoothspline import smoothspline
from scipy import io

class ProDenICA:
    def __init__(self):
        self.A =None
        self.s =None
    def orth(self,A):
        u,d,vh =np.linalg.svd(A)
        return u@vh
    def centering(self,X):
        X =X.copy()
        X =X-np.mean(X,axis=0)[np.newaxis,:]
        return X
    def whitening(self,X):
        cov =(X.T@X)/X.shape[0]
        u,d,vh =np.linalg.svd(cov)
        whiten_matrix =np.diag(1.0/np.sqrt(d))@np.linalg.inv(u)
        X =(whiten_matrix@X.T).T
        return [whiten_matrix,X]
    def amari(self,A0,A):
        p =A.shape[1]
        r =A0@np.linalg.inv(A)
        r_abs =np.abs(r)
        row_abssum =np.sum(r_abs,axis=1)
        row_absmax =np.max(r_abs,axis=1)
        col_abssum =np.sum(r_abs,axis=0)
        col_absmax =np.sum(r_abs,axis=0)
        val =1.0/(2*p)*(np.sum((row_abssum/row_absmax)-1.0)+np.sum((col_abssum/col_absmax)-1.0))
        return val
    def mixmat(self,p):
        A0 =np.random.normal(loc=0,scale=1.0,size=(p,p))
        u,d,vh =np.linalg.svd(A0)
        d =np.random.uniform(low=0.0,high=1.0,size=(p,))+1.0
        A0 =u@np.diag(d)@vh
        return A0
    def expements(self,s):
        [N,p] =s.shape
        print('s shape ',s.shape)
        A0=self.mixmat(p)
        print('A0  shape ',A0.shape)
        print('A0 ',A0)
        X =(A0@s.T).T
        print('X shape ',X.shape)
        X =self.centering(X)
        [whiten_matrix,X] =self.whitening(X)
        W0 =np.linalg.inv(A0)@np.linalg.inv(whiten_matrix)
        return [W0,X]
    def grouping(self,s,L):
        sl =np.zeros((L,),dtype=float)
        yl =np.zeros((L,),dtype=float)
        s =np.sort(s)
        step =((s[-1]-s[0])/(L-1))*1.01
        left  =s[0]-0.5*step
        for i in range(L):
            sl[i] =left +(2*i+1)*step/2
        for i in range(s.shape[0]):
            yl[int(np.round((s[i]-left)/step))] =yl[int(np.round((s[i]-left)/step))]+1.0
        yl =yl/s.shape[0]
        return [sl,yl,step]
    def gauss(self,x):
        return (1.0/(np.sqrt(2*np.pi)))*np.exp(-1.0*x*x/2.0)
        
    def ica(self,X,df=6,itmax=20,L=1000,tol=1e-4):
        [N,p]=X.shape
        X =self.centering(X)
        [whiten_matrix,X]=self.whitening(X)
        W =np.random.normal(loc=0,scale=1.0,size=(p,p))
        W =self.orth(W)
        W_last =np.zeros((p,p),dtype=float)
        s =np.zeros((N,p),dtype=float)
        sl =np.zeros((L,p),dtype=float)
        wl =np.zeros((L,p),dtype=float)
        zl =np.zeros((L,p),dtype=float)
        yl =np.zeros((L,p),dtype=float)
        ul =np.zeros((L,p),dtype=float)
        it =0
        sp=[]
        s =X@W
        for i in range(p):
            sp1 =smoothspline()
            sp.append(sp1)
            
        
        while(it<itmax):
            ## g() stage
            print('it= ',it,' dist(W,W_last): ',np.linalg.norm(np.abs(W.T@W_last)-np.eye(p)))
            W_last =W.copy()
            s =X@W
            for i in range(p):
                [sl[:,i],yl[:,i],step] =self.grouping(s[:,i],L=L)
                gg12 =np.ones((L,),dtype=float)
                gg1  =np.zeros((L,),dtype=float)
                while(np.linalg.norm(gg12-gg1)>1e-4):
                    print("norm(g-g_last) ",np.linalg.norm(gg12-gg1))
                    gg12 =gg1.copy()
                    gsl =self.gauss(sl[:,i])
                    ul[:,i] =np.exp(gg1)*gsl
                    wl[:,i] =ul[:,i]
                    zl[:,i] =((yl[:,i]-ul[:,i])/(ul[:,i])) +gg1
                    sp[i].fit(x=sl[:,i],y=zl[:,i],df=df,w=wl[:,i],tol=tol)
                    gg1 =sp[i].predict(sl[:,i])[:,0]
                
                
                    
            
            ## W stage
            
            for i in range(p):
                print('sp[',i,'] ',sp[i])
                gg2 =sp[i].predict(s[:,i])
                W[:,i] =np.mean(np.multiply(X,gg2[:,1][:,np.newaxis]),axis=0)-np.mean(gg2[:,2])*W[:,i]
                W[:,i] =W[:,i]/np.linalg.norm(W[:,i])
            W =self.orth(W)
            it =it +1
        
        plt.subplots(figsize=(12,6))
        
        for i in range(p):
            plt.subplot(p,1,i+1)
            lines =np.linspace(np.min(s[:,i]),np.max(s[:,i]),2000)
            yp =sp[i].predict(lines)[:,0]
            plt.plot(lines,yp,'-b',label="pdf"+str(i))
            plt.legend()
            
        
        return [W,s]
                
                
                
            
     

        
np.random.seed(0)
n_samples = 500
p =2 
time = np.linspace(0, 8, n_samples)



s1 = np.random.laplace(loc=0,scale=1.0,size=(n_samples,))
s2 = np.random.uniform(low=-1.0,high=1.0,size=(n_samples,))




S =np.c_[s1,s2]
S =S -np.mean(S,axis=0)

ICA =ProDenICA()
[W0,X] =ICA.expements(s=S)
[W,y]=ICA.ica(X,itmax=10,df=10)

print('amarti score ',ICA.amari(W0,W))
plt.subplots(figsize=(12,6))
for i in range(p):
    plt.subplot(p,1,i+1)
    plt.plot(np.linspace(0,n_samples,n_samples),S[:,i])



plt.subplots(figsize=(12,6))
for i in range(p):
    plt.subplot(p,1,i+1)
    plt.plot(np.linspace(0,n_samples,n_samples),y[:,i])







    
        
        
        

        
        
        
        