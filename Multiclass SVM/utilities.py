import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
import itertools as it


def Polinomial_kernel(X, Y, gamma):
    K = np.power(np.dot(X.T, Y) + 1, gamma)
    return K

def Gaussian_kernel(X,Y,gamma):
    diff = np.zeros(len(X[0])*len(Y[0])).reshape(len(X[0]),len(Y[0]))
    for i in range(len(X)):
        diff += np.square(X[i].reshape(-1, 1)- Y[i].reshape(1, -1))
    norms = np.sqrt(diff)
    Phi = np.exp(- gamma *((norms) ** 2))
    return Phi


def minimize(X_Train, Y_Train, C, gamma, kern):
    
    X = X_Train.T
    Y = np.dot(Y_Train.reshape(-1,1),Y_Train.reshape(1,-1))
    
    e = -1 * np.ones((X.shape[1],1))
    
    if kern == 'gaussian':
        kernel = Gaussian_kernel(X, X, gamma)
    if kern == 'polynomial':
        kernel = Polinomial_kernel(X, X, gamma)
    
    
    Pi = np.multiply(kernel,Y).astype('float')
    P = matrix(Pi)
    
    q = matrix(e, tc='d')
    
    A = Y_Train.reshape(1,-1).astype('float') 
    A = matrix(A)
    
    b = np.array([0]).astype('float')
    b = matrix(b)
    
    G = np.concatenate((np.identity(X.shape[1]), -1 * np.identity(X.shape[1]))).astype('float')
    G = matrix(G)
    
    h = np.concatenate((C * np.ones((X.shape[1],1)), np.zeros((X.shape[1],1)))).astype('float')
    h = matrix(h)
    
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    sol = solvers.qp(P,q, G, h, A, b)
    lamda = np.array(sol['x'])
    #print('lamda', lamda)
    b_opt = []
    for i in range(lamda.shape[0]):
        if lamda[i] > 1e-5 and lamda[i] < C - 1e-5:
            b = (1 - Y_Train[i]*np.dot(kernel[:,i].reshape(1,-1),np.multiply(Y_Train,lamda))) / Y_Train[i]
            b_opt.append(b)
    b_opt = np.array(b_opt).reshape(-1,1)
    if b_opt.shape[0] >= 1:
        b_opt = np.mean(b_opt)
    else:
        b_opt = 0 
    
    #print('e', e.shape)
    gradient = np.dot(Pi,lamda)+ e
    #print('grad', gradient)
    
    R=np.where(np.logical_or(np.logical_and(lamda.reshape(-1)<C -1e-5,Y_Train.reshape(-1)>0),np.logical_and(lamda.reshape(-1)> 1e-5,Y_Train.reshape(-1)<0)))[0]
    S=np.where(np.logical_or(np.logical_and(lamda.reshape(-1)<C -1e-5,Y_Train.reshape(-1)<0),np.logical_and(lamda.reshape(-1)> 1e-5,Y_Train.reshape(-1)>0)))[0]
    m_alpha = max(-1 * (gradient[R] / Y_Train[R]))
    #print('vectr_m', -1 * (gradient[R] / Y_Train[R]))
    #print('vectr_M', -1 * (gradient[S] / Y_Train[S]))
    M_alpha = min(-1 * (gradient[S] / Y_Train[S]))

    print("m_alpha - M_alpha",m_alpha - M_alpha)
    #time.sleep(1)
    return sol, b_opt, lamda, kernel


def dec_fun(Y_Train, lamda, kernel, b_opt):
    arg = (np.sum(Y_Train.reshape(-1,1) * lamda * kernel, axis=0)
                    + b_opt).reshape(-1,1)
    return arg