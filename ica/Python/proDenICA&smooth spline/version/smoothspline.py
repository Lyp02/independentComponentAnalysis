#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 23:36:27 2020

@author: lyp
"""
import numpy as np
import matplotlib.pyplot as plt

class smoothspline:
    def __init__(self):
        self.ux =None
        self.uy =None
        self.uw =None
        self.p =None
        self.df=None
        self.lbd=None
        self.fpx =None
        self.ppx =None
        self.K =None
    def fit(self,x,y,df,w,tol=1e-4):
  
        n =x.shape[0]
        """
        ux as knots should be unique
        """
        #print('x ',x)
        indexs =np.argsort(x)
  
        x1 =x[indexs].copy()
        y1 =y[indexs].copy()
        w1 =w[indexs].copy()
 
        nonu,indics =np.unique(np.round((x1-np.mean(x1))/tol),return_index=True)
        nx =indics.shape[0]
        if(nx<4):
            print('the unique knots are not enough')
            return
        ux =np.zeros((nx,),dtype=float)
        uy =np.zeros((nx,),dtype=float)
        uw =np.zeros((nx,),dtype=float)
        for i in range(nx-1):
            ux[i] =x1[indics[i]]
            uy[i] =np.mean(y1[indics[i]:indics[i+1]])
            uw[i] =np.mean(w1[indics[i]:indics[i+1]])
        ux[nx-1] =x1[indics[nx-1]]
        uy[nx-1] =np.mean(y1[indics[nx-1]:n])
        uw[nx-1] =np.mean(w1[indics[nx-1]:n])
        self.ux =ux
        self.uy =uy
        self.uw =uw
        c =np.zeros((nx-2,1),dtype=float)
        u =np.zeros((nx-2,1),dtype=float)
        a =np.zeros((nx,1),dtype=float)
        R =np.zeros((nx-2,nx-2),dtype=float)
        Qt=np.zeros((nx-2,nx),dtype=float)
        

        dx =ux[1:nx]-ux[0:nx-1]
        for i in range(nx-2):
            if(i==0):
                R[i,0]=2*(dx[0]+dx[1])
                R[i,1]=dx[1]
            elif(i==nx-3):
                R[nx-3,nx-4]=dx[nx-3]
                R[nx-3,nx-3]=2*(dx[nx-3]+dx[nx-2])
            else:
                R[i,i-1]   =dx[i]
                R[i,i]     =2*(dx[i]+dx[i+1])
                R[i,i+1]   =dx[i+1]
            Qt[i,i]   =1.0/dx[i]
            Qt[i,i+1] =(-1.0/dx[i])-(1.0/dx[i+1])
            Qt[i,i+2] =1.0/dx[i+1]
            
            
        K =Qt.T@np.linalg.inv(R)@Qt
        e_vals,e_vecs =np.linalg.eig(K)
        lbd_last=0
        lbd=1.0
        while(np.abs(lbd_last-lbd)>1e-7):
            lbd_last=lbd
            dg =df -np.sum(1.0/(1+lbd*e_vals))
            hg =np.sum((1.0/(np.square(1+lbd*e_vals)))*e_vals)
            lbd =lbd -dg/hg
        #print('lbd ',lbd)
            
        p =1.0/(1.0+lbd)
        self.p =p
        self.lbd=lbd
        self.df =df
        self.K =K
        D2 =np.diag(1.0/uw)
        
        u =np.linalg.inv(6*(1-p)*Qt@D2@Qt.T+p*R)@(Qt@uy[:,np.newaxis])
        c =3*p*u
        a =uy[:,np.newaxis]-6*(1-p)*D2@Qt.T@u
        da =a[1:nx,0]-a[0:nx-1,0]
        fpx =np.zeros((nx,4),dtype=float)
        fpx[:,0] =a[:,0]
        fpx[1:nx-1,2] =6*p*u[:,0]
        fpx[:nx-1,3] =(fpx[1:nx,2]-fpx[0:nx-1,2])/dx
        fpx[:nx-1,1] =da/dx -(fpx[:nx-1,2]/2)*dx-(fpx[:nx-1,3]/6)*np.square(dx)
        fpx[nx-1,1] =fpx[nx-2,1]+fpx[nx-2,2]*dx[nx-2]+0.5*fpx[nx-2,3]*dx[nx-2]*dx[nx-2]
        self.fpx =fpx
        return
        
    def predict(self,x):
        """
        x increasing order
        """
        n =x.shape[0]
        ppx =np.zeros((n,3),dtype=float)
        i =0
        while(i<n and x[i]<self.ux[0]):
            dx =x[i]-self.ux[0]
            ppx[i,0] =self.fpx[0,0]+self.fpx[0,1]*(dx)
            ppx[i,1] =self.fpx[0,1]
            i =i +1
        low =i
        i =n-1
        while(i>=0 and x[i]>=self.ux[-1]):
            dx =x[i]-self.ux[-1]
            ppx[i,0] =self.fpx[-1,0]+self.fpx[-1,1]*(dx)
            ppx[i,1] =self.fpx[-1,1]
            i =i -1
        high =i
        k =0
        i =low
        nx =self.ux.shape[0]
        while(i<=high):
            while(k<nx and x[i]>=self.ux[k+1]):
                k =k +1
            dx =x[i]-self.ux[k]
            ppx[i,0]=self.fpx[k,0]+self.fpx[k,1]*dx+0.5*self.fpx[k,2]*dx*dx+(1.0/6)*self.fpx[k,3]*dx*dx*dx
            ppx[i,1]=self.fpx[k,1]+self.fpx[k,2]*dx+0.5*self.fpx[k,3]*dx*dx
            ppx[i,2]=self.fpx[k,2]+self.fpx[k,3]*dx
            i =i +1
        
        self.ppx =ppx
        return ppx
        
        
        
                
        
"""
x =np.linspace(-20,20,1000)
y =0.5*x*x-2*x+12*np.cos(x)+np.sin(0.4*x)
y1 =y +np.random.normal(loc=0,scale=1.0,size=(1000,))
w =np.ones((1000,),dtype=float)
lbd =100
sp =smoothspline()
sp.fit(x=x,y=y1,df=40,w=w)
y2 =sp.predict(x=x)
plt.subplots(figsize=(12,6))
plt.plot(x,y,'-o',label='true')
plt.plot(x,y2[:,0],'-r',label='sp')
plt.legend()       
"""           
            
            
            
        
        
        
        