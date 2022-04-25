import scipy.io as sio
from sklearn import svm
from sklearn import metrics

spam_train = sio.loadmat('spamTrain.mat')
spam_test = sio.loadmat('spamTest.mat')

train_X, train_y = spam_train.get('X'), spam_train.get('y').ravel()
print(train_X.shape, train_y.shape)
test_X, test_y = spam_test.get('Xtest'), spam_test.get('ytest').ravel()
print(test_X.shape, test_y.shape)

svc = svm.SVC()
svc.fit(train_X, train_y)
print(metrics.classification_report(test_y, svc.predict(test_X)))
