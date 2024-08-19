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


gamma = 0.2
C = 2
t0 = time.time()
results = minimize(X_Train, Y_Train, C, gamma, 'gaussian')
print('Time', time.time() - t0)


_, cMat, acc = train_predict(X_Train, Y_Train, gamma, results)

print('Train accuracy:', acc)

disp = ConfusionMatrixDisplay(confusion_matrix=cMat, display_labels=['Q','O'])
disp.plot()

_, cMat, acc = test_predict(X_Train, Y_Train, X_Test,Y_Test, gamma, results)

print('Test accuracy:', acc)

disp = ConfusionMatrixDisplay(confusion_matrix=cMat, display_labels=['Q','O'])
disp.plot()