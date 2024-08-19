from utilities import *

Q_O_data = pd.read_csv('Letters_Q_O.csv')

D_data = pd.read_csv('Letter_D.csv')

data = [Q_O_data, D_data]
data = pd.concat(data)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(data[:,0:16], data[:,-1].reshape(-1,1),
                                                    test_size=0.2, random_state=42, stratify=data[:,-1].reshape(-1,1))


# Normalization: first fit the scaler on X training samples then transform them and the X test samples
scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform(X_Test)


dataset = np.hstack((X_Train, Y_Train))


letters = ['Q','O','D']
results = {}
C = 5
gamma = 0.5
X_Train_base  = X_Train
Y_Train_base = Y_Train
for couple in list(it.combinations(letters,2)):
    
    # SPLIT THE DATASET
    l = list(set(letters) - set(couple))[0]
    print('coppia', couple, 'l', l)
    mask = (dataset[:,-1] != l)
    d_set = dataset[mask,:]
    #print('data shape', d_set.shape)
    X_Train , Y_Train = d_set[:,0:16].astype('float32'), d_set[:,-1].reshape(-1,1)
    #print('Train shape', X_Train.shape, 'Test shape', Y_Train.shape)
    # Encode the variable letter: LETTER = 1 ; NOT LETTER = -1
    letter = list(couple)[0]
    Y_Train = np.where(Y_Train == letter, 1, -1).astype('float32')
    #print('letter', letter)
    #print('train', type(X_Train))
    #print('test', type(X_Test))
    # SOLVE
    sol, b_opt, lamda, kernel = minimize(X_Train, Y_Train, C, gamma, 'gaussian')
    results[letter + list(couple)[1] + 'b_opt'] = b_opt
    results[letter + list(couple)[1] + 'lamda'] = lamda
    results[letter + list(couple)[1] + 'kernel'] = kernel
    results[letter + list(couple)[1] + 'Y_Train'] = Y_Train
    results[letter + list(couple)[1] + 'X_Train'] = X_Train
    
    Y_Train = Y_Train_base 

    args = []
args.append(dec_fun(results['QOY_Train'], results['QOlamda'],
                    Gaussian_kernel(results['QOX_Train'].T, X_Train_base.T, gamma), results['QOb_opt']))
args.append(dec_fun(results['QDY_Train'], results['QDlamda'],
                    Gaussian_kernel(results['QDX_Train'].T, X_Train_base.T, gamma), results['QDb_opt']))
args.append(dec_fun(results['ODY_Train'], results['ODlamda'],
                    Gaussian_kernel(results['ODX_Train'].T, X_Train_base.T, gamma), results['ODb_opt']))

y_pred = []
for elem in range(Y_Train_base.shape[0]):
    votes = []
    res = []
    res.append(args[0][elem])
    res.append(args[1][elem])
    res.append(args[2][elem])
    # as Q
    vote = 0.5 * (res[0] + 1 + res[1] + 1)
    votes.append(vote)
    # as O
    vote = 0.5 * (-res[0] + 1 + res[2] + 1)
    votes.append(vote)
    # as D
    vote = 0.5 * (-res[1] + 1 - res[2] + 1)
    votes.append(vote)  
    #print(votes)
    idx = np.argmax(np.array(votes)) 
    #print(idx)
    #print(letters[idx])
    if idx.shape == 1:
        y_pred.append(letters[idx])
    else:
        y_pred.append(letters[np.min(idx)])
    
TrainPred = np.array(y_pred).reshape(-1,1)    
print('Train accuracy', accuracy_score(Y_Train_base.reshape(-1,1), TrainPred))

ConMatrixTrain = confusion_matrix(Y_Train.reshape(-1,1), TrainPred)

disp = ConfusionMatrixDisplay(confusion_matrix=ConMatrixTrain,display_labels=['D','O','Q'])
disp.plot()

args = []
args.append(dec_fun(results['QOY_Train'], results['QOlamda'],
                    Gaussian_kernel(results['QOX_Train'].T, X_Test.T, gamma), results['QOb_opt']))
args.append(dec_fun(results['QDY_Train'], results['QDlamda'],
                    Gaussian_kernel(results['QDX_Train'].T, X_Test.T, gamma), results['QDb_opt']))
args.append(dec_fun(results['ODY_Train'], results['ODlamda'],
                    Gaussian_kernel(results['ODX_Train'].T, X_Test.T, gamma), results['ODb_opt']))

y_pred = []
for elem in range(Y_Test.shape[0]):
    votes = []
    res = []
    res.append(args[0][elem])
    res.append(args[1][elem])
    res.append(args[2][elem])
    # as Q
    vote = 0.5 * (res[0] + 1 + res[1] + 1)
    votes.append(vote)
    # as O
    vote = 0.5 * (-res[0] + 1 + res[2] + 1)
    votes.append(vote)
    # as D
    vote = 0.5 * (-res[1] + 1 - res[2] + 1)
    votes.append(vote)  
    #print(votes)
    idx = np.argmax(np.array(votes)) 
    #print(idx)
    #print(letters[idx])
    if idx.shape == 1:
        y_pred.append(letters[idx])
    else:
        y_pred.append(letters[np.min(idx)])             
    
TestPred = np.array(y_pred).reshape(-1,1)    
print('Test accuracy',accuracy_score(Y_Test.reshape(-1,1), TestPred))

ConMatrixTest = confusion_matrix(Y_Test.reshape(-1,1), TestPred)

disp = ConfusionMatrixDisplay(confusion_matrix=ConMatrixTest,display_labels=['D','O','Q'])
disp.plot()

