import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from cvxopt import matrix, solvers
solvers.options["show_progress"]=False
import time

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

def minimize(X_Train, Y_Train, C, gamma, kern):
    #INPUT:
    #X_Train: matrix of dimension P*16
    #Y_Train: array of dimension P*1
    #C and gamma are the hyperparamters chosen
    #kern is a string, that can be either gaussian or polynomial
    #the output of the function is a dictionary, containing the results given by
    #the optimization solver, the optimal values of b and lamda (the
    #dual variables) and the gaussian kernel matrix
    
    
    X = X_Train.T
    Y = np.dot(Y_Train.reshape(-1,1),Y_Train.reshape(1,-1))
    e = -1 * np.ones((X.shape[1],1))
    if kern == 'gaussian':
        kernel = Gaussian_kernel(X, X, gamma)
    if kern == 'polynomial':
        kernel = Polinomial_kernel(X, X, gamma)
    
    
    #we build the matrices that the solver cvxopt requires
    
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
    
    # we set the tolerance of the solver
    solvers.options['abstol'] = 1e-12
    solvers.options['reltol'] = 1e-12
    t0=time.time()
    sol = solvers.qp(P,q, G, h, A, b)
    iters=sol["iterations"]
    time_optimization = time.time()-t0
    lamda = np.array(sol['x'])
    #computing of b
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
    #updating of the gradient and computing of kkt_viol
    gradient = np.dot(Pi,lamda)+ e
    R=np.where(np.logical_or(np.logical_and(lamda.reshape(-1)<C -1e-5,Y_Train.reshape(-1)>0),np.logical_and(lamda.reshape(-1)> 1e-5,Y_Train.reshape(-1)<0)))[0]
    S=np.where(np.logical_or(np.logical_and(lamda.reshape(-1)<C -1e-5,Y_Train.reshape(-1)<0),np.logical_and(lamda.reshape(-1)> 1e-5,Y_Train.reshape(-1)>0)))[0]
    m_alpha = max(-1 * (gradient[R] / Y_Train[R]))
    M_alpha = min(-1 * (gradient[S] / Y_Train[S]))
    kkt_viol=m_alpha- M_alpha
    
    #computing the value of the objective function
    
    f_obj = 0.5 * np.dot(np.dot(lamda.reshape(-1,1).T, Pi), lamda.reshape(-1,1)) + np.dot(e.T, lamda.reshape(-1,1))
    print('f_obj', f_obj)
    
    result = {'sol': sol, 'b_opt': b_opt, 'lamda': lamda, 'kernel': kernel, 'kern': kern,
              "optimization time":time_optimization,"number iterations":iters,"KKT violation":kkt_viol,"objective function":f_obj}    
    return result

def train_predict(X_Train, Y_Train, gamma, results):  
    #INPUT:
    #X_Train: matrix of dimension P*16
    #Y_Train: array of dimension P*1
    #gamma is the hyperparameter chosen
    #results is the dictionary, output of the minimize function.
    # OUTPUT:
    # pred: training predictions
    # ConMatrix: the confusion matrix
    # acc: training accuracy 
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
    #INPUT:
    #X_Train: matrix of dimension P*16
    #Y_Train: array of dimension P*1
    #X_Test: matrix of dimension P_test*16
    #Y_Test: array of dimension P_test*1
    #gamma is the hyperparameter chosen
    #results is the dictionary, output of the minimize function.
    # OUTPUT:
    # pred: test predictions
    # ConMatrix: the confusion matrix
    # acc: test accuracy 
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

def k_folds(X_Train, Y_Train, k=6):
    # INPUT:
    #X_Train: matrix of dimension P*16
    #Y_Train: array of dimension P*1
    #k is the number of fold we want to create (default 6)
    # OUTPUT:
    #list_folds: a list containing X_train, Y_train, X_val, Y_val for each fold
    if X_Train.shape[0] % k != 0:
        print(f'Is impossible to divide the training set in {k} folds')
    else:    
        kf = StratifiedKFold(n_splits=k)
        list_folds = []
        for train_index, val_index in kf.split(X_Train, Y_Train):
            X_train, X_val = X_Train[train_index,:], X_Train[val_index,:]
            Y_train, Y_val = Y_Train[train_index], Y_Train[val_index]
            single_list = [X_train, X_val, Y_train, Y_val]
            list_folds.append(single_list)
        return list_folds
    
def GridSearchCV(folds, kern):
    # INPUT:
    #fold: output of the function k_folds()
    #kern is a string, that can be either gaussian or polynomial
    # OUTPUT:
    #best_hyp: a dictionary containing C and gamma optimal, the average
    # training accuracy and the average validation accuracy
    gamma_list = [0.2] #[i/100 for i in range(10,30,1)]
    C_list = [5]#[i/10 for i in range(10,61,1)]
    n_k = len(folds)
    res_opt = 0
    bound = 0
    best_hyp = {'gamma': 0, 'C': 0, 'val': 0, 'train': 0}
    for gamma in gamma_list:
        for C in C_list:
            tot_val_acc = 0
            tot_train_acc = 0
            for k in folds:
                    X_train, X_val, Y_train, Y_val = k[0], k[1], k[2], k[3]
                    print('X_val', X_val.shape)
                    res = minimize(X_train, Y_train, C, gamma, kern)
                    _,_,v_acc = test_predict(X_train, Y_train ,X_val, Y_val, gamma, res)
                    _,_,t_acc = train_predict(X_train, Y_train, gamma, res)
                    tot_val_acc += v_acc
                    tot_train_acc += t_acc
            avg_val_acc = tot_val_acc / n_k    
            avg_train_acc = tot_train_acc / n_k        
            print(f"Gamma: {gamma}; C: {C}")
            print("Average training Accuracy:", avg_train_acc)            
            print("Average validation Accuracy:", avg_val_acc)
            print('--------------------------------------------------')
            if avg_val_acc > bound:
                        bound = avg_val_acc
                        best_hyp['gamma'], best_hyp['C'] = gamma, C
                        best_hyp['val'], best_hyp['train'] = avg_val_acc, avg_train_acc
    return best_hyp  


