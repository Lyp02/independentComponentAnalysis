#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:50:24 2020

@author: lyp
"""

def FastICA_Decorreation4(X,M=40,L=60,maxiters=800,choice='none',tol=1e-14,ratio=0.9995):
    """
    并行 FastICA盲源反卷积 
                         (1)SVD not converge 是内存问题？，求解pinv时
                         (2)M,ratio 

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
        W2 =np.ones((p*M,p))
        k =-1
        while(np.linalg.norm(W2-W)>tol):
            if(k==20):
                break
            W2 =W.copy()
            k = k +1
            print('k= ',k,' ||W-W2|| ',np.linalg.norm(W2-W))
            for i in range(p):
                w =W[:,i].copy()
                mask[:]=0.0
                mask[i*(2*L+1):(i+1)*(2*L+1)]=1.0
                index =np.where(mask==0.0)[0]
                
                B1 =B[:,index].copy()                   
                e_vals,e_vecs =np.linalg.eig(B1@B1.T)
                indexs_sort =np.argsort(-1.0*e_vals)
                e_vals2 =e_vals[indexs_sort]
                print('e_vals , ',e_vals2)
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
                
                
            
                print('B1 rank ',np.linalg.matrix_rank(B1))
                #print('Q rank ',np.linalg.matrix_rank(Q))
                beta =np.linalg.pinv(B1)@(w[:,np.newaxis])
                #print('beta shape ',beta.shape)
                print('before ',np.linalg.norm(w))
                w =w -(B1@beta)[:,0]  #保证矩阵B1是超定的，最小二乘求其正交补
                print('after ',np.linalg.norm(w))
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
    """
 
        
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
    
    return [W,y,beta,Y]
        
            