from utilities import *

data = pd.read_csv('Letters_Q_O.csv')

# Encode the variable letter: Q = 1 ; O = -1
data["letter"] = np.where(data["letter"].str.contains("Q"), 1, -1)

# Split in Train set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(data[:,0:16], data[:,-1].reshape(-1,1),
                                                    test_size=0.2, random_state=42, stratify=data[:,-1].reshape(-1,1))

# Normalization: first fit the scaler on X training samples then transform them and the X test samples
scaler = StandardScaler().fit(X_Train)
X_Train = scaler.transform(X_Train)
X_Test = scaler.transform(X_Test)

X_Test = X_Test.T

folds = k_folds(X_Train, Y_Train,6) 

best_hyp = GridSearchCV(folds, 'gaussian')

C = best_hyp['C']
gamma =  best_hyp['gamma']
results = minimize(X_Train, Y_Train, C, gamma, 'gaussian')


sol=results["sol"]
b_opt=results["b_opt"]
lamda=results["lamda"]
kernel=results["kernel"]

# DECISION FUNCTION
TrainPred = np.sign(np.sum(Y_Train.reshape(-1,1) * lamda * kernel, axis=0)
                    + b_opt).reshape(-1,1)

ConMatrixTrain = confusion_matrix(Y_Train.reshape(-1,1), TrainPred)

disp = ConfusionMatrixDisplay(confusion_matrix=ConMatrixTrain, display_labels=['Q','O'])
disp.plot(cmap="Greens")


X = X_Train.T

# DECISION FUNCTION
TestPred = np.sign(np.sum(Y_Train.reshape(-1,1) * lamda * Gaussian_kernel(X, X_Test, gamma), axis=0)
                   + b_opt).reshape(-1,1)

ConMatrixTest = confusion_matrix(Y_Test.reshape(-1,1), TestPred)

disp = ConfusionMatrixDisplay(confusion_matrix=ConMatrixTest, display_labels=['Q','O'])
disp.plot()

