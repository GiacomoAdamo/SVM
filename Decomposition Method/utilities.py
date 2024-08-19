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


def minimize(X_Train, Y_Train, C, gamma, kern,qu):#qu is the dimension of the working set
    t0=time.time()
    X = X_Train.T
    X=np.array(X)
    dimension=X_Train.shape[0]
    indexes=np.arange(dimension)
    e = -1 * np.ones((X.shape[1],1))
    lamda=np.zeros(dimension)

    k = 0
    gradient=-1 * np.ones((dimension,1))
    M_alpha = -1
    m_alpha = 1
    #m_alpha - M_alpha >-0.001 or
    R=np.where(Y_Train>0)[0]
    S=np.where(Y_Train<0)[0]
    while  m_alpha - M_alpha > 1e-5:
        k=k+1
        print("                                                 ",k )
        lamda_previous_iteration = np.copy(lamda)
        W=[]
        i=np.argsort((+gradient[R]*Y_Train[R]).reshape(-1))[0:int(qu/2)]
        j=np.argsort((-gradient[S]*Y_Train[S]).reshape(-1))[0:int(qu/2)]
        W=np.concatenate((np.array(R)[list(i)],np.array(S)[list(j)]))
        W=list(W)
          
        list_difference = list(set(indexes).difference(set(W)))
        Kernel_q = Gaussian_kernel(X[:,W], X[:,W], gamma)
        Y_ridotta =np.dot(Y_Train[W].reshape(-1,1),Y_Train[W].reshape(1,-1))
        P = np.multiply(Y_ridotta,Kernel_q)
        P = matrix(P)
        e = -1 * np.ones((Kernel_q.shape[1],1))
        matr=Gaussian_kernel(X[:,list_difference], X[:,W], gamma)
        y_list_difference=np.dot(Y_Train[list_difference].reshape(-1,1),Y_Train[W].reshape(1,-1))
        matr2=np.multiply(matr,y_list_difference)
        lamda_list_difference = lamda[list_difference].reshape(-1,1)
        q = matrix(np.dot(matr2.T,lamda_list_difference) + e, tc='d')
        A = Y_Train[W].reshape(1,-1).astype(np.double) 
        A = matrix(A)
        b = matrix(-np.dot(Y_Train[list_difference].reshape(1,-1),lamda[list_difference]),tc='d')
        G = np.concatenate((np.identity(Kernel_q.shape[1]), -1 * np.identity(Kernel_q.shape[1]))).astype(np.double)
        G = matrix(G)
        h = np.concatenate((C * np.ones((Kernel_q.shape[1],1)), np.zeros((Kernel_q.shape[1],1)))).astype(np.double)
        h = matrix(h)
        
        solvers.options['show_progess'] = False        
        solvers.options['abstol'] = 1e-10
        solvers.options['reltol'] = 1e-10
        sol = solvers.qp(P, q, G, h, A, b)
        lamda_ristretto = np.array(sol['x'])
        update = np.zeros(dimension)
        for j in range(len(W)):
            lamda[W[j]]=lamda_ristretto[j]

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
    
    result = { 'b_opt': b_opt, 'lamda': lamda,'kern':kern}    
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