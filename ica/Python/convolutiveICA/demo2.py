#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 02:53:34 2020

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
    eps =1e-12
    a =np.exp(x)
    b =np.exp(-1.0*x)
    return ((a-b)/(a+b+eps))
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
    
    


def FastICA_Decorreation(X,M=40,L=60,maxiters=200,choice='none',tol=1e-9,ratios=0.9999995):
    """
    串行 FastICA盲源反卷积
                 
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
    #W =np.random.normal(loc=0,scale=1.0,size=(p*M,p))
    W =np.zeros((p*M,p),dtype=float)
    
    for i in range(p):
        W[i*M,i]=1.0
    w =np.zeros((p*M,),dtype=float)
    w_last =np.zeros((p*M,),dtype=float)
    B =np.zeros((p*M,p*(2*L+1)),dtype=float)
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
    
    
    
    for i in range(p):
        k =-1
        w =W[:,i].copy()
        w_last =np.random.normal(loc=0,scale=1.0,size=(p*M,))
        print('i = ',i)
        while(k<maxiters):
            if(abs(abs(np.sum(w_last*w))-1.0)<tol):  #看论文不认真，停止条件都写错了，到底是多少错误，这么简单的东西，
                                                     #照着文章去弄，没有下限。
                break
            #print('i= ',i,' k= ',k,'lim ',abs(abs(np.sum(w_last*w))-1.0))
            k =k+1
            
            w_last =w.copy()
            y =XX@(w_last[:,np.newaxis])
            y1 =np.multiply(XX,g(y,choice=choice,order=1))
            y2 =g(y[:,0],choice=choice,order=2)
            w  =((np.mean(y1,axis=0)[:,np.newaxis]))[:,0]-np.mean(y2)*w_last
            w  =w/np.linalg.norm(w)
            
            """
            正交化 避免找到相同信号 -->归一化
            要用正规的投影方法，是针对欠定矩阵，自己这里QR分解简直胡来，返回单位矩阵这个Q也是对的
            如果是得到特征值，也不能保证是正交的。
            怎么获得矩阵列空间的表达。--->自己许多做法简直是胡来，怎么可能获得想要的结论。
            """
            if(i>0):                                       
                #[Q,_] =np.linalg.qr(B[:,0:(i*(2*L+1))])  #Q是满秩的，再做正交化有什么意义呢？-->怎么正交
                #print('before ',np.linalg.norm(w))       #随着L约束的增加，这里是趋向于满秩。--->似乎不大行
                #w = w -(Q@Q.T@w[:,np.newaxis])[:,0]
                #print('after ',np.linalg.norm(w))
                #print('B rank ',np.linalg.matrix_rank(B[:,0:(i*(2*L+1))]))
                #print('Q rank ',np.linalg.matrix_rank(Q))
                
                B1 =B[:,0:(i*(2*L+1))].copy()                     #做特征值分解，选取主分量。这里不是看矩阵是否满秩。
                                                                  #与选取列空间是一样的吗。--->是一样的
                e_vals,e_vecs =np.linalg.eig(B1@B1.T)             
            
                indexs_sort =np.argsort(-1.0*e_vals)
                e_vals2 =e_vals[indexs_sort]
 
                e_vecs2 =e_vecs[:,indexs_sort]
                base =np.sum(e_vals2)
                t =0.0
                          #ratio 似乎不论怎么设置，最后选取的特征向量个数都是很小，难道必须强制数量，而不是门限

                counts =-1
                for j in range(p*M):
                    if((np.sqrt(t/base)>=ratios[i-1])):
                        break
                    else:
                        t =t +e_vals2[j]
                        counts =j
                print('p*M ',p*M,' counts ',counts)
                B1 =e_vecs2[:,0:counts+1]                      #B1是正交的啊，根本不需要用最小二乘方法。避免求逆
                
                
                
                beta =np.linalg.pinv(B1)@(w[:,np.newaxis])
                #print('beta shape ',beta.shape)
                #print('before ',np.linalg.norm(w))
                w =w -(B1@beta)[:,0]                   #保证矩阵B1是超定的，最小二乘求其正交补.i=2时刻几乎减去所有成分。
                #print('after ',np.linalg.norm(w))     #L的增大，使得i=2时刻几乎减去所有成分，看来方法不行
                """
                if(i==2):
                    print((B[:,0:(i*(2*L+1))].T@w[:,np.newaxis]))   #归一化之后，感觉不相关做的不好，
                 """                                                                     #数值比较大。并且corr(1,2)小，但是corr(2,1)大
            w  =w/np.linalg.norm(w)
        
        
        """
        print('IC ',i)
        if(i>0):
            print(B[:,0:i*(2*L+1)].T@(w[:,np.newaxis]))
        """
        W[:,i]=w.copy()
        """
        投影空间填充
        """
        
        for j in range(2*L+1):
            B[:,i*(2*L+1)+j] = (Rs[j,:,:]@(w[:,np.newaxis]))[:,0]
    
    """
    for i in range(p):
        for j in range(p):
            if(i!=j):
                print('i= ',i,' j= ',j)
                print(B[:,i*(2*L+1):(i+1)*(2*L+1)].T@(W[:,j][:,np.newaxis]))
    """
    y =XX@(W)
    [beta,Y] = reconstruct(X=X,U=y[L:N+L,:],R=L)  #现在至少波形比较相似了，太好了，
    
    
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
            
    
    #/pics/FastICA_deflation/
    
    return [W,y,beta,Y]    
   



def FastICA_Decorreation4(X,M=40,L=60,maxiters=800,choice='none',tol=1e-14,ratio=0.9995):
    """
    并行 FastICA盲源反卷积 
                         (1)SVD not converge 是内存问题？，求解pinv时,g() overflow

    """
    [N,p]=X.shape
    """
    构造输入矩阵-->中心化-->白化
    """
    Xs =np.zeros((N+2*M,p),dtype=float)
    XX =np.zeros((N+2*L,p*M),dtype=float)
    print('X shape ',X.shape)
    print('Xs shape ',Xs.shape)
    Xs[M:N+M,:]=centralization(X)[1]
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
        #print('W ',W)
        #print('W_last ',W_last)
        print('it= ',it ,'dist(W,W_last)',np.linalg.norm(np.abs(W_last.T@W)-np.eye(p)))
        
        W_last =W.copy()
        for i in range(p):
            w =W[:,i].copy()
            w_last =w.copy()
            y =XX@(w_last[:,np.newaxis])
            y1 =np.multiply(XX,g(y,choice=choice,order=1))
            #print('y1 ',y1)
            y2 =g(y[:,0],choice=choice,order=2)
            #print('y2 ',y2)
            #print('before ','w',i,' :')
            #print(w)
            #print('np.mean(y2)*w_last ',np.mean(y2))
            #print('((np.mean(y1,axis=0)[:,np.newaxis]))[:,0] ',((np.mean(y1,axis=0)[:,np.newaxis]))[:,0])
            w  =((np.mean(y1,axis=0)[:,np.newaxis]))[:,0]-np.mean(y2)*w_last
            #print('after ','w',i,' :')
            #print(w)
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
                #print('before ','w',i,' :')
                #print(w)
                w =w -(B1@B1.T@w[:,np.newaxis])[:,0]
                w  =w/np.linalg.norm(w)
                #print('after ','w',i,' :')
                #print(w)
               
                for j in range(2*L+1):
                    B[:,i*(2*L+1)+j] = (Rs[j,:,:]@(w[:,np.newaxis]))[:,0]
                W[:,i]=w.copy()
        
        
    
    
    y =XX@(W) 
    

    [beta,Y] = reconstruct(X=X,U=y[L:N+L,:],R=L)  #现在至少波形比较相似了，太好了，
    
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
    print('w shape ',W.shape)
    print('whitenMatrix shape ',whitenMatrix.shape)
    print('W@whitenMatrxi ',W.T@whitenMatrix)
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
sources  =io.loadmat('simulation_sources.mat')['sources']
mixtures =io.loadmat('simulation_mixtures.mat')['mixtures']



#m=12 L=20

M=12
L =400
X =mixtures
[N,p]=X.shape
[A_,S_,beta_,Y_] = FastICA_Decorreation4(X,M=M,L=L,maxiters=400,choice='sub',tol=1e-9,ratio=0.99)


print('y1 y2 corre ',np.corrcoef(S_[:,0],S_[:,1])[0,1])


plt.subplots(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(np.arange(X.shape[0]),X[:,0],'-b',label=r"$x_{1}$")
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(X.shape[0]),X[:,1],'-b',label=r"$x_{2}$")
plt.legend()
#fig = plt.gcf()
#fig.savefig("./2x2 symmetric mode(simulations)/observations.pdf")


plt.subplots(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(np.arange(sources.shape[0]),sources[:,1],'-b',label=r"$s_{1}$")
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(sources.shape[0]),sources[:,0],'-b',label=r"$s_{2}$")
plt.legend()
#fig = plt.gcf()
#fig.savefig("./2x2 symmetric mode(simulations)/sources.pdf")


#fig = plt.gcf()
#fig.savefig("original signal.pdf")

plt.subplots(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(np.arange(S_[:,0].shape[0]),S_[:,1],'-b',label=r"$y_{1}$")
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(S_[:,1].shape[0]),S_[:,0],'-b',label=r"$y_{2}$")
plt.legend()
#fig = plt.gcf()
#fig.savefig("./2x2 symmetric mode(simulations)/innovation process.pdf")

plt.subplots(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(np.arange(Y_.shape[1]),Y_[1,:,0],'-b',label=r"$\hat{s}_{11}$")
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(Y_.shape[1]),Y_[0,:,0],'-b',label=r"$\hat{s}_{21}$")
plt.legend()
#fig = plt.gcf()
#fig.savefig("./2x2 symmetric mode(simulations)/recovered sources.pdf")


        
plt.subplots(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(np.arange(sources.shape[0]),sources[:,0],'-b',label=r"$s_{1}$")
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(S_[:,0].shape[0]),-1.0*S_[:,0],'-b',label=r"$y_{1}$")
plt.legend()

#fig = plt.gcf()
#fig.savefig("./2x2 symmetric mode/sources.pdf")


#fig = plt.gcf()
#fig.savefig("original signal.pdf")

plt.subplots(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(np.arange(sources.shape[0]),sources[:,1],'-b',label=r"$s_{2}$")
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(S_[:,1].shape[0]),S_[:,1],'-b',label=r"$y_{2}$")
plt.legend()

