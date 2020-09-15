#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 05:29:37 2020

@author: lyp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Latex
from scipy import stats 
from scipy import signal
from scipy import interpolate
import seaborn as sns
import matplotlib as mpl
from matplotlib.font_manager import fontManager  
import os 
import lasio
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy import io
from scipy import linalg
from sklearn.decomposition import FastICA, PCA





"""
辅助函数
"""
def sigmoid(x):
    return  np.exp(x)/(1.0+np.exp(x))
def tanh(x):
    a =np.exp(x)
    b =np.exp(-1.0*x)
    return ((a-b)/(a+b))
def cosh(x):
    a =np.exp(x)
    b =np.exp(-1.0*x)
    return (a+b)/2
def sign(x):
    if(x>0):
        return 1.0
    else:
        return -1.0

def removewave(X,ref,M=50,tol=1e-3):
    [N,p] =X.shape
    #print('N ',N,' p ',p )
    y =np.zeros((N+2*M,p),dtype=float)
    y[M:N+M,:] =X.copy()
    Xc =np.zeros((N+2*M,2*M+1),dtype=float)
    #print('Xc,shape ',Xc.shape)
    #print('ref.shape ',ref.shape)
    for i in range(2*M+1):
        Xc[i:N+i,i] =ref
    corrs =np.zeros((p,2*M+1),dtype=float)
    for i in range(p):
        corrs[i,:] =np.sum(np.multiply(Xc,y[:,i][:,np.newaxis]),axis=0)/np.sum(Xc*Xc,axis=0)
        max_val =np.max(abs(corrs[i,:]))
        #print('corrs max ',max_val)
        index   =np.argmax(abs(corrs[i,:]))
        while(max_val>tol):
            #print('max_val ',max_val)
            y[:,i] =y[:,i] -(corrs[i,index]*Xc[:,index])
            corrs[i,:] =np.sum(np.multiply(Xc,y[:,i][:,np.newaxis]),axis=0)/np.sum(Xc*Xc,axis=0)
            max_val =np.max(abs(corrs[i,:]))
            index   =np.argmax(abs(corrs[i,:]))
            break
    #plt.subplots(figsize=(12,6))
    #plt.plot(np.linspace(0,N,N),ref,'-b')
    #plt.subplots(figsize=(12,6))
    #for i in range(p):
    #    plt.subplot(p,2,i+1)
    #    plt.plot(np.linspace(0,N,N),X[:,i],'-b')
    #    plt.subplot(p,2,p+i+1)
    #    plt.plot(np.linspace(0,N,N),y[M:N+M,i],'-g')
    return y[M:N+M,:]

def centralization(X):
    mean =np.mean(X,axis=0)[np.newaxis,:]
    return [mean,X-mean]

def whiten(X):  
    [N,p] =X.shape
    w,v =np.linalg.eig(X.T@X/N)
    B =np.diag(1.0/np.sqrt(w))@v.T
    X1 =(B@X.T).T
    #print(np.linalg.det(X1.T@X1/N))
    return X1

def whiten2(X):  
    [N,p] =X.shape
    w,v =np.linalg.eig(X.T@X/N)
    whitenMatrix =np.diag(1.0/np.sqrt(w))@v.T
    dewhitenMatrix =v@np.diag(np.sqrt(w))
    X1 =(whitenMatrix@X.T).T
    #print(np.linalg.det(X1.T@X1/N))
    return [X1,whitenMatrix,dewhitenMatrix]

def g(x,choice,order,eps=1e-14):
    if(order==0):
        if(choice=="sup"):
            a2=1.000
            return (-1.0/a2)*np.exp(-1.0*a2*x*x/2)
        elif(choice=='sub'):
            t =x*x
            return (1.0/4)*t*t
        else:
            a1=1.5
            return (1.0/a1)*np.log(cosh(a1*x))  #之前是写错了。
    elif(order==1):
        if(choice=='sup'):
            a2 =1.000
            return x*np.exp(-1.0*a2*(x*x)/2)
        elif(choice=='sub'):
            return x*x*x
        else:
            a1=1.5
            return tanh(a1*x)
    elif(order==2):
        if(choice=='sup'):
            a2=1.005
            t =np.exp(-1.0*a2*(x*x)/2)
            return t-a2*x*x*t
        elif(choice=='sub'):
            return 3*x*x
        else:
            a1 =1.5
            return a1*(1-np.square(tanh(a1*x)))
    else:
        print('err')
        
        
def reconstruct(X,U,R=50):
    [N,p] =U.shape
    X1 =np.zeros((N+2*R,p),dtype=float)
    X1[R:N+R,:]=X
    Us =np.zeros((p,N+2*R,2*R+1),dtype=float)
    for i in range(p):
        for j in range(2*R+1):
            Us[i,j:N+j,j] =U[:,i]
    beta =np.zeros((p,2*R+1,p),dtype=float)
    Y    =np.zeros((p,N+2*R,p),dtype=float)
    for i in range(p):
        beta[i] =np.linalg.pinv(Us[i,:,:])@(X1)
        Y[i,:] =(Us[i,:,:]@beta[i])
    return [beta,Y]

def get_delay(X,U):
    [N,p] =X.shape
    
    
    



def FastICA_Decorreation4(X,M=40,L=60,maxiters=800,choice='none',tol=1e-14,ratio=0.9995):
    """
    并行 FastICA盲源反卷积 
                         (1)SVD not converge 是内存问题？，求解pinv时

    """
    [N,p]=X.shape
    """
    构造输入矩阵-->中心化-->白化
    """
    Xs =np.zeros((N+2*M,p),dtype=float)
    XX =np.zeros((N+2*L,p*M),dtype=float)
    Xs[M:N+M,:]=X.copy()
    for i in range(N):
        for j in range(p):
            XX[L+i,j*M:(j+1)*M] =np.flip(Xs[i+1:i+M+1,j])
    [mean,XX] =centralization(XX)
    [XX,whitenMatrix,dewhitenMatrix] =whiten2(XX)
    print('whitenMatrix shape ',whitenMatrix.shape)
    print('whitenMatrix rank ',np.linalg.matrix_rank(whitenMatrix))
    #W =np.random.normal(loc=0,scale=1.0,size=(p*M,p))  #能不能的到想要的结果和这个W的初始化十分相关。
    #W =np.zeros((p*M,p),dtype=float)
    """
    W 的初始化，得到结果受初始化影响
    """
    W =np.zeros((p*M,p),dtype=float)
    for i in range(p):
        W[i*M,i]=1.0
    W =(W.T@np.linalg.inv(whitenMatrix)).T
    
    
  
    W_last =np.zeros((p*M,p))
    W2 =np.ones((p*M,p))
    w =np.zeros((p*M,),dtype=float)
    w_last =np.zeros((p*M,),dtype=float)
    B =np.zeros((p*M,p*(2*L+1)),dtype=float)
    mask =np.zeros((p*(2*L+1),),dtype=float)
    """
    计算相关矩阵R_l -L<=l<=L
    """
    Rs =np.zeros((2*L+1,p*M,p*M),dtype=float)
    for i in range(N):
        for j in range(2*L+1):
            Rs[j,:,:] =Rs[j,:,:] +(XX[L+i,:][:,np.newaxis])@(XX[L+i+j-L,:][np.newaxis,:])
    Rs =Rs/N
    

    
    """
    FastICA 主程序
    """
    
    
    it =-1

    while(it<maxiters):
        it =it+1
        print('it= ',it ,'dist(W,W_last)',np.linalg.norm(np.abs(W_last.T@W)-np.eye(p)))
        W_last =W.copy()
        for i in range(p):
            w =W[:,i].copy()
            w_last =w.copy()
            y =XX@(w_last[:,np.newaxis])
            y1 =np.multiply(XX,g(y,choice=choice,order=1))
            y2 =g(y[:,0],choice=choice,order=2)
            w  =((np.mean(y1,axis=0)[:,np.newaxis]))[:,0]-np.mean(y2)*w_last
            w  =w/np.linalg.norm(w)
            W[:,i]=w.copy()

            """
            投影空间填充
            """
            for j in range(2*L+1):
                B[:,i*(2*L+1)+j] = (Rs[j,:,:]@(w[:,np.newaxis]))[:,0]
        
        """
        去除相关
        """
       
        k =-1
        W2[:,:]=0
        while(np.linalg.norm(W2-W)>tol):
            print('k= ',k,' ||W-W2|| ',np.linalg.norm(W2-W))
            W2 =W.copy()
            k = k +1
            if(k>200):
                break
            for i in range(p):
                w =W[:,i].copy()
                mask[:]=0.0
                mask[i*(2*L+1):(i+1)*(2*L+1)]=1.0
                index =np.where(mask==0.0)[0]
                
                B1 =B[:,index].copy()                   
                e_vals,e_vecs =np.linalg.eig(B1@B1.T)
                indexs_sort =np.argsort(-1.0*e_vals)
                e_vals2 =e_vals[indexs_sort]
                #print('e_vals , ',e_vals2)
                e_vecs2 =e_vecs[:,indexs_sort]
                base =np.sum(e_vals2)
                t =0.0
                counts =-1
                for j in range(p*M):
                    if(np.sqrt(t/base)>=ratio):
                            break
                    else:
                        t =t +e_vals2[j]
                        counts =j
                print('p*M ',p*M,' counts ',counts)
                B1 =e_vecs2[:,0:counts+1]                       
                
                
            
                #print('B1 rank ',np.linalg.matrix_rank(B1))
                #print('Q rank ',np.linalg.matrix_rank(Q))
                #beta =linalg.pinv(B1)@(w[:,np.newaxis])
                #print('beta shape ',beta.shape)
                #print('before ',np.linalg.norm(w))
                #w =w -(B1@beta)[:,0]  #保证矩阵B1是超定的，最小二乘求其正交补
                #print('after ',np.linalg.norm(w))
                w =w -(B1@B1.T@w[:,np.newaxis])[:,0]
                w  =w/np.linalg.norm(w)
               
                for j in range(2*L+1):
                    B[:,i*(2*L+1)+j] = (Rs[j,:,:]@(w[:,np.newaxis]))[:,0]
                W[:,i]=w.copy()
        
        
    
    
    y =XX@(W) 
    

    [beta,Y] = reconstruct(X=X,U=y[L:N+L,:],R=M)  #现在至少波形比较相似了，太好了，
    
    """
    for i in range(p):
        plt.subplots(figsize=(12,6))
        for j in range(p):
            plt.subplot(p,1,j+1)
            
            plt.plot(np.arange(beta.shape[1]),beta[i,:,j],label='contributions : '+'u'+str(i)+'on'+'x'+str(j))
            plt.legend()
    #plt.suptitle('contributions')
    
    #plt.subplots(figsize=(12,6))
    for i in range(p):
        plt.subplots(figsize=(12,6))
        for j in range(p):
            plt.subplot(p,1,j+1)
            
            plt.plot(np.arange(Y.shape[1]),Y[i,:,j],label='componenets: '+'u'+str(i)+'on'+'x'+str(j))
            plt.legend()
    #plt.suptitle('components')
    
    #计算延迟
    
    for i in range(p):
        plt.subplots(figsize=(12,6))
        a =beta[i,:,i]
        index =-1
        for j in range(p):
            b =beta[i,:,j]
            c =np.correlate(a,b,mode='full')
            plt.subplot(p,1,j+1)
            plt.plot(np.arange(c.shape[0]),c,'-o',label='d'+str(i)+str(j))
            plt.legend()
            print('d'+str(i)+str(j))
            print(np.argmax(abs(c))-a.shape[0]+1)
    """
    return [W,y,beta,Y]
        
            




"""
读取仿真数据
"""
data=io.loadmat('data.mat')['data']
label=io.loadmat('label.mat')['label']
slowness=io.loadmat('slowness.mat')['slowness']

sim=io.loadmat('sim_data.mat')['sim']
length=io.loadmat('sim_data_lens.mat')['length']
waves=io.loadmat('waves.mat')['wave']
dataS =io.loadmat('sources.mat')['sources']





n_samples =1000

#simulation of  acoustic wave 
s1 =sim[60,0,0:n_samples]
s2 =sim[30,2,0:n_samples]
s3 =sim[20,1,0:n_samples]

#s2 =np.zeros((n_samples,),dtype=float)
#s3 =np.zeros((n_samples,),dtype=float)
#s2[0:(n_samples//30)] =np.random.laplace(loc=0,scale=3.0,size=(n_samples//30,))
#s3[0:(n_samples//10)] =np.random.uniform(low=-4,high=4,size=(n_samples//10))

#"""
#speech  music data
n_samples =8000
s1 =dataS[:,0]
s2 =dataS[:,1]
s3 =dataS[:,2]
#"""

"""
#simulation 
time = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * time) # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time)) # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time) # Signal 3: saw tooth signal
"""





time = np.linspace(0, 8, n_samples)

d1 =2
d2 =7
d3 =12




d1=2
d2=12
d3=12


delay =np.zeros((3,3),dtype=int)
delay[0][0]=0;delay[0][1]=d2;delay[0][2]=d3;
delay[1][1]=0;delay[1][0]=d1;delay[1][2]=2*d3
delay[2][2]=0;delay[2][0]=2*d1;delay[2][1]=2*d2

s1 =(s1-np.mean(s1))/np.std(s1)

s2 =(s2-np.mean(s2))/np.std(s2)
s3 =(s3-np.mean(s3))/np.std(s3)

x1 =np.zeros((100+n_samples,),dtype=float)
x2 =np.zeros((100+n_samples,),dtype=float)
x3 =np.zeros((100+n_samples,),dtype=float)

xx1 =np.zeros((100+n_samples,),dtype=float)
xx2 =np.zeros((100+n_samples,),dtype=float)
xx3 =np.zeros((100+n_samples,),dtype=float)

x11 =np.zeros((100+n_samples,),dtype=float)
x22 =np.zeros((100+n_samples,),dtype=float)
x33 =np.zeros((100+n_samples,),dtype=float)

ss1 =np.zeros((2*n_samples,),dtype=float)
ss2 =np.zeros((2*n_samples,),dtype=float)
ss3 =np.zeros((2*n_samples,),dtype=float)

#"""
#标准的bss问题
x1[50:(n_samples+50)] =1.0*s1
x1[(30+d2):(30+d2+n_samples)] =0.6*s2 +x1[(30+d2):(30+d2+n_samples)]
x1[(70+d3):(70+d3+n_samples)] =0.7*s3 +x1[(70+d3):(70+d3+n_samples)]
x2[30:(n_samples+30)] =1.0*s2
x2[(50+d1):(50+d1+n_samples)]=0.6*s1 +x2[(50+d1):(50+d1+n_samples)]
x2[(70+2*d3):(70+2*d3+n_samples)] =0.7*s3 +x2[(70+2*d3):(70+2*d3+n_samples)]
x3[70:(n_samples+70)] =1.0*s3
x3[(50+2*d1):(50+2*d1+n_samples)]=0.6*s1 +x3[(50+2*d1):(50+2*d1+n_samples)]
x3[(30+2*d2):(30+2*d2+n_samples)] =0.7*s2 +x3[(30+2*d2):(30+2*d2+n_samples)]
X_sim =np.c_[x1,x2,x3].copy()

xx1[50:(n_samples+50)] =1.0*s1
xx1[(30+d2):(30+d2+n_samples)] =1.0*s2 +xx1[(30+d2):(30+d2+n_samples)]
xx2[30:(n_samples+30)] =1.0*s2
xx2[(50+d1):(50+d1+n_samples)]=1.0*s1 +xx2[(50+d1):(50+d1+n_samples)]
#xx2[(70+d3):(70+d3+n_samples)] =1.0*s3 +xx2[(70+d3):(70+d3+n_samples)]

#xx3[70:(n_samples+70)] =1.0*s3
#xx3[(30+d2):(30+d2+n_samples)] =1.0*s2 +xx3[(30+d2):(30+d2+n_samples)]
X_sim2 =np.c_[xx1,xx2].copy()
#"""








"""
#阵列声波问题 如果当作是瞬时的话，那么混合矩阵不可逆，方法失效。
#当作卷积时，矩阵是可逆的在d1<d2<d3条件下，Vandermonde矩阵。 
x1[50:(n_samples+50)] =1.0*s1
x1[(30):(30+n_samples)] =1.0*s2 +x1[(30):(30+n_samples)]
x1[(70):(70+n_samples)] =1.0*s3 +x1[(70):(70+n_samples)]
x2[30+d2:(n_samples+30+d2)] =0.6*s2
x2[(50+d1):(50+d1+n_samples)]=0.4*s1 +x2[(50+d1):(50+d1+n_samples)]
x2[(70+d3):(70+d3+n_samples)] =0.5*s3 +x2[(70+d3):(70+d3+n_samples)]
x3[70+2*d3:(n_samples+70+2*d3)] =0.7*s3
x3[(50+2*d1):(50+2*d1+n_samples)]=0.6*s1 +x3[(50+2*d1):(50+2*d1+n_samples)]
x3[(30+2*d2):(30+2*d2+n_samples)] =0.3*s2 +x3[(30+2*d2):(30+2*d2+n_samples)]


delay[0][0]=0;delay[0][1]=0;delay[0][2]=0;
delay[1][1]=d2;delay[1][0]=d1;delay[1][2]=d3
delay[2][2]=2*d3;delay[2][0]=2*d1;delay[2][1]=2*d2
"""

x11[(50):(50+n_samples)]=s1
x22[(50+d1):(50+d1+n_samples)]=s2
x33[(50+2*d1):(50+2*d1+n_samples)]=s3 





 
#[N,p] =X_sim.shape


"""
X_sim2 =np.c_[data[26,:,0],data[26,:,1],data[26,:,2]]
X_sim  =np.zeros((X_sim2.shape[0]//6,X_sim2.shape[1]),dtype=float)
for i in range(X_sim.shape[0]):
    X_sim[i,:]=X_sim2[i*6,:]
X_real =np.c_[waves[26,:,0],waves[26,:,1],waves[26,:,2]]  
#X =(X-np.mean(X,axis=0)[np.newaxis,:])/np.std(X,axis=0)[np.newaxis,:]
"""
X_real =np.zeros((750,3),dtype=float)
X_real[50:(50+660),:] =np.c_[waves[26,:,0],waves[26,:,1],waves[26,:,2]]

#S =np.c_[s1,s2,s3]
#io.savemat('./dataS.mat', {'dataS':S})
S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)
S/= S.std(axis=0)

A= np.array([[1, 0.2, 0.6], [0.5, 1.0, 1.5], [0.6, 2.0, 1.0]])  #混合矩阵必须是可逆的FastICA才可以，阵列声波这样好像不行。
X_ins= np.dot(S, A.T)





M=50
L =150
X =X_sim
[N,p]=X.shape
[A_,S_,beta_,Y_] = FastICA_Decorreation4(X,M=M,L=L,maxiters=400,choice='none',tol=1e-9,ratio=0.9999995)


print('y1 y2 corre ',np.corrcoef(S_[:,0],S_[:,1])[0,1])
print('y1 y3 corre ',np.corrcoef(S_[:,0],S_[:,2])[0,1])
print('y2 y3 corre ',np.corrcoef(S_[:,1],S_[:,2])[0,1])


plt.subplots(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(np.arange(X.shape[0]),X[:,0],'-b',label=r"$x_{1}$")
plt.legend()
plt.subplot(3,1,2)
plt.plot(np.arange(X.shape[0]),X[:,1],'-b',label=r"$x_{2}$")
plt.legend()
plt.subplot(3,1,3)
plt.plot(np.arange(X.shape[0]),X[:,2],'-b',label=r"$x_{3}$")
plt.legend()
#fig = plt.gcf()
#fig.savefig("./2x2 symmetric mode/observations.pdf")


plt.subplots(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(np.arange(s1.shape[0]),s1,'-b',label=r"$s_{1}$")
plt.legend()
plt.subplot(3,1,2)
plt.plot(np.arange(s2.shape[0]),s2,'-b',label=r"$s_{2}$")
plt.legend()
plt.subplot(3,1,3)
plt.plot(np.arange(s2.shape[0]),s3,'-b',label=r"$s_{3}$")
plt.legend()
#fig = plt.gcf()
#fig.savefig("./2x2 symmetric mode/sources.pdf")


#fig = plt.gcf()
#fig.savefig("original signal.pdf")

plt.subplots(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(np.arange(S_[:,0].shape[0]),S_[:,0],'-b',label=r"$y_{1}$")
plt.legend()
plt.subplot(3,1,2)
plt.plot(np.arange(S_[:,1].shape[0]),S_[:,1],'-b',label=r"$y_{2}$")
plt.legend()
plt.subplot(3,1,3)
plt.plot(np.arange(S_[:,2].shape[0]),S_[:,2],'-b',label=r"$y_{3}$")
plt.legend()
#fig = plt.gcf()
#fig.savefig("./2x2 symmetric mode/innovation process.pdf")

plt.subplots(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(np.arange(Y_.shape[1]),Y_[0,:,0],'-b',label=r"$\hat{s}_{11}$")
plt.legend()
plt.subplot(3,1,2)
plt.plot(np.arange(Y_.shape[1]),Y_[1,:,0],'-b',label=r"$\hat{s}_{21}$")
plt.legend()
plt.subplot(3,1,3)
plt.plot(np.arange(Y_.shape[1]),Y_[2,:,0],'-b',label=r"$\hat{s}_{31}$")
plt.legend()
#fig = plt.gcf()
#fig.savefig("./2x2 symmetric mode/recovered sources.pdf")