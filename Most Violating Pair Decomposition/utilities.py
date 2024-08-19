import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
import random, time
np.random.seed(1802731)

def Polinomial_kernel(X, Y, gamma):
    K = np.power(np.dot(X.T, Y) + 1, gamma)
    return K

def Gaussian_kernel(X,Y,gamma):
    #this function takes in input two matrices X,Y. The matrix X has dimension 16*A
    # while the matrix Y has dimension 16*B. The function gives as an output the gaussian 
    #kernel,which is a matrix of dimension A*B
    
    diff = np.zeros(len(X[0])*len(Y[0])).reshape(len(X[0]),len(Y[0]))
    for i in range(len(X)):
        diff += np.square(X[i].reshape(-1, 1)- Y[i].reshape(1, -1))
    norms = np.sqrt(diff)
    Phi = np.exp(- gamma *((norms) ** 2))
    return Phi


def kernel_q(K,W):#W is a list containing the indexes of the working set
    List=[]
    for i in W:
        List.append(K[i,W])
    return np.array(List)



def minimize(X_Train, Y_Train, C, gamma, kern):#qu is the dimension of the working set
    X = X_Train.T
    dimension=X_Train.shape[0]
    indexes=np.arange(dimension)
    e = -1 * np.ones((X.shape[1],1))
    lamda=np.zeros(dimension)
    
    if kern == 'gaussian':
        kernel = Gaussian_kernel(X, X, gamma)
     
    k = 0
    
    gradient=-1 * np.ones((dimension,1))
    R=list(np.where(Y_Train>0)[0])
    S=list(np.where(Y_Train<0)[0])
    m_alpha =1 
    M_alpha = -1
    
    while  m_alpha - M_alpha > 1e-5:
        k=k+1
        print("                                                 ",k )
        lamda_previous_iteration = np.copy(lamda)
        W0 = [np.argmax(-1 * gradient[R] * Y_Train[R]),np.argmin(-1 * gradient[S] * Y_Train[S])]
        i=R[W0[0]]
        j=S[W0[1]]
        W=[i,j]
        print("W",W)
        list_difference = list(set(indexes).difference(set(W)))
        Kernel_q = Gaussian_kernel(X[:,W], X[:,W], gamma)
        Y_ridotta =np.dot(Y_Train[W].reshape(-1,1),Y_Train[W].reshape(1,-1))
        
        Q_ridotta =np.multiply(Y_ridotta,Kernel_q)
        
        
        lamda_ristretto=np.copy(lamda[W])
        d=np.array([Y_Train[i],-Y_Train[j]]).reshape(-1,1)
        product_gradient_direction=np.dot(gradient[W].T,d).reshape(-1)
    
        if float(product_gradient_direction)<=1e-5 and float(product_gradient_direction)>=-1e-5:#1e-3:
            beta_star=0
            d_star=0
        else:
            if product_gradient_direction<-1e-5:
                d_star=d
            else:
                d_star=-d
         #compute beta
            if d[0]>0 and d[1]>0:
                beta=min(C-lamda[i],C-lamda[j])
            
            if d[0]<0 and d[1]<0:
                beta=min(lamda[i],lamda[j])
            
            if d[0]>0 and d[1]<0:
                beta=min(C-lamda[i],lamda[j])
        
            if d[0]<0 and d[1]>0:
                beta=min(lamda[i],C-lamda[j])
        
            if beta<= 1e-5:
                beta_star=0
            else:
                if np.dot(np.dot(d_star.reshape(1,-1),Q_ridotta),d_star.reshape(-1,1))< 1e-5:
                    beta_star=beta
                if np.dot(np.dot(d_star.reshape(1,-1),Q_ridotta),d_star.reshape(-1,1))>= 1e-5:
                    beta_nv= -(np.dot(gradient[W].reshape(1,-1),d_star.reshape(-1,1)))/(np.dot(np.dot(d_star.reshape(1,-1),Q_ridotta),d_star.reshape(-1,1)))
                    beta_star=min(beta,beta_nv)
        
        
        lamda_ristretto=lamda_ristretto.reshape(-1,1) + beta_star*d_star.reshape(-1,1)
        
        for o in range(len(W)):
            lamda[W[o]]=np.squeeze(lamda_ristretto[o])
        
        columns=Gaussian_kernel(X,X[:,W],gamma)
        y_3=np.dot(Y_Train.reshape(-1,1),Y_Train[W].reshape(1,-1))
        
        columns=np.multiply(columns,y_3)
        gradient = gradient + np.dot(columns,lamda[W].reshape(-1,1)-lamda_previous_iteration[W].reshape(-1,1))           
        
        R=np.where(np.logical_or(np.logical_and(lamda<C -1e-5,Y_Train.reshape(-1)>0),np.logical_and(lamda> 1e-5,Y_Train.reshape(-1)<0)))[0]
        S=np.where(np.logical_or(np.logical_and(lamda<C -1e-5,Y_Train.reshape(-1)<0),np.logical_and(lamda> 1e-5,Y_Train.reshape(-1)>0)))[0]
        
        m_alpha = max(-1 * (gradient[R] / Y_Train[R]))
        M_alpha = min(-1 * (gradient[S] / Y_Train[S]))
        print("m_alpha - M_alpha",m_alpha - M_alpha)
            
            
        
 
    b_opt = []
    lamda = lamda.reshape(-1,1)
    free_SV=np.where(np.logical_and(lamda.reshape(-1)<C -1e-5,lamda.reshape(-1)>1e-5))[0]
    kernel_SV=Gaussian_kernel(X,X[:,free_SV],gamma)
    for i in range(len(list(free_SV))):
        b = (1 - Y_Train[free_SV[i]]*np.dot(kernel_SV[:,i].reshape(1,-1),np.multiply(Y_Train,lamda))) / Y_Train[free_SV[i]]
        b_opt.append(b)
    b_opt = np.array(b_opt).reshape(-1,1)
    if b_opt.shape[0] >=1:
        b_opt = np.mean(b_opt)
    else:
        b_opt = 0    
    print('b', b_opt)    
    result = {'b_opt': b_opt, 'lamda': lamda, 'kernel': kernel, 'kern': kern}    
    return result

def train_predict(X_Train, Y_Train, gamma, results):    
    X = X_Train.T
    kern, lamda, b_opt = results['kern'], results['lamda'], results['b_opt']
    
    if kern == 'gaussian':
        kernel = Gaussian_kernel(X, X, gamma)
    if kern == 'polynomial':
        kernel = Polinomial_kernel(X, X, gamma)
        
    # DECISION FUNCTION
    pred = np.sign(np.sum(Y_Train.reshape(-1,1) * lamda * kernel, axis=0)
                    + b_opt).reshape(-1,1)
    
    #Confusion matrix 
    ConMatrix = confusion_matrix(Y_Train.reshape(-1,1), pred)
    #Accuracy
    acc= accuracy_score(Y_Train.reshape(-1,1), pred)
    
    return pred, ConMatrix, acc


def test_predict(X_Train, Y_Train ,X_Test, Y_Test, gamma, results):    
    X = X_Train.T
    kern, lamda, b_opt = results['kern'], results['lamda'], results['b_opt']
    
    if kern == 'gaussian':
        kernel = Gaussian_kernel(X, X_Test.T, gamma)
    if kern == 'polynomial':
        kernel = Polinomial_kernel(X, X_Test.T, gamma)
        
    # DECISION FUNCTION
    pred = np.sign(np.sum(Y_Train.reshape(-1,1) * lamda * kernel, axis=0)
                    + b_opt).reshape(-1,1)
    
    #Confusion matrix 
    ConMatrix = confusion_matrix(Y_Test.reshape(-1,1), pred)
    #Accuracy
    acc = accuracy_score(Y_Test.reshape(-1,1), pred)
    
    return pred, ConMatrix, acc