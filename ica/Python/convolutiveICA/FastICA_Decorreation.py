#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 04:58:34 2020

@author: lyp
"""

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
                if(k==100):
                    print('i= ',i)
                    print('e_vals , ',e_vals2)
                e_vecs2 =e_vecs[:,indexs_sort]
                base =np.sum(e_vals2)
                t =0.0
                          #ratio 似乎不论怎么设置，最后选取的特征向量个数都是很小，难道必须强制数量，而不是门限

                counts =-1
                for j in range(p*M):
                    if((np.sqrt(t/base)>=ratios)):
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
    [beta,Y] = reconstruct(X=X,U=y[L:N+L,:],R=M)  #现在至少波形比较相似了，太好了，
    
    
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