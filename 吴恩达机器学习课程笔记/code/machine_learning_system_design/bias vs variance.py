import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def cost(theta, X, y):
    m = X.shape[0]
    square_sum = (X.dot(theta) - y).T.dot(X.dot(theta) - y)
    return (1 / (2 * m)) * square_sum

def regularized_cost(theta, X, y, _lambda=1):
    m = X.shape[0]
    regularized_term = theta[1 :].T.dot(theta[1 :])
    return cost(theta, X, y) + (_lambda / (2 * m)) * regularized_term

def gradient(theta, X, y):
    m = X.shape[0]
    return (X.T.dot(X.dot(theta) - y)) / m

def regularized_gradient(theta, X, y, _lambda=1):
    m = X.shape[0]
    regularized_term = theta.copy()
    regularized_term[0] = 0
    return gradient(theta, X, y) + (_lambda / m) * regularized_term


def batch_gradient_decent(theta, X, y, epoch, alpha=0.01, _lambda=1):
    cost_data = [regularized_cost(theta, X, y, _lambda)]
    _theta = theta.copy()
    for i in range(epoch):
        _theta = _theta - alpha * regularized_gradient(_theta, X, y, _lambda)
        cost_data.append(regularized_cost(_theta, X, y, _lambda))
    return _theta, np.array(cost_data).reshape((len(cost_data), 1))

def polynomial_features(x, power):
    data = {}
    for i in range(1, power + 1):
        data["f{}".format(i)] = np.power(x[:, 0], i)
    return pd.DataFrame(data)

def normalize_feature(df):
    for i in range(df.shape[1]):
        df.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].mean()) / df.iloc[:, i].std()
    return df


#1.1 Visualizing the dataset
data = sio.loadmat('ex5data1.mat')
X = np.array(data['X'])
y = np.array(data['y'])
plt.scatter(X, y, label='Training data')
plt.xlabel('Change in water level(x)')
plt.ylabel('Water flowing out of the dam(y)')
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
print(X.shape, y.shape)

#Regularized linear regression cost function
theta = np.ones((X.shape[1], 1))
print(theta.shape)
print(regularized_cost(theta, X, y))

#Regularized linear regression gradient
print(regularized_gradient(theta, X, y))

#Fitting linear regression
epoch = 5000
alpha = 0.001
_lambda = 1

final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha, _lambda)
print(final_theta)

b = final_theta[0]
m = final_theta[1]

plot_X = np.arange(-50, 40, 1)

plt.plot(plot_X, plot_X * m + b, label="Prediction")
plt.legend(loc=2)

#2 Bias-variance
#2.1 Learning curves

train_cost, cv_cost = [], []
Xval = np.array(data['Xval'])
yval = np.array(data['yval'])
Xval = np.concatenate([np.ones((Xval.shape[0], 1)), Xval], axis=1)
print(Xval.shape, yval.shape)

m = X.shape[0]
for i in range(1, m + 1):
    final_theta, _ = batch_gradient_decent(theta, X[: i,:], y[: i,:], epoch, alpha, 0)
    train_cost.append(cost(final_theta, X[: i,:], y[: i,:]))
    cv_cost.append(cost(final_theta, Xval, yval))
train_cost = np.array(train_cost).reshape(len(train_cost), 1)
cv_cost = np.array(cv_cost).reshape(len(cv_cost), 1)

plt.figure()
error = np.arange(1, m + 1).reshape((m, 1))
plt.plot(error, train_cost, label='Train')
plt.plot(error, cv_cost, label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend(loc='best')


#3 Ploynomial regression

#3.1 learning Ploynomial Regression
#3.2 Adjusting the regularization parameter
X = np.array(data['X'])

df = polynomial_features(X, 3)
df_n = normalize_feature(df)
X = np.concatenate([np.ones((X.shape[0], 1)), np.array(df_n)], axis=1)
print(X.shape)
theta = np.ones((X.shape[1], 1))
epoch = 5000
alpha = 0.01
_lambda = 0
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha, _lambda)
print(cost_data[-1])
print(final_theta)
plt.figure()
plt.scatter(data['X'], y)
plot_X = np.arange(-45, 45)
plot_X = plot_X.reshape((plot_X.shape[0], 1))
plot_X_p = np.array(normalize_feature(polynomial_features(plot_X, 3)))
plot_X_p = np.concatenate([np.ones((plot_X_p.shape[0], 1)), plot_X_p], axis=1)
plt.plot(plot_X, plot_X_p.dot(final_theta))

Xval, yval = np.array(data['Xval']), np.array(data['yval'])

train_cost, cv_cost = [], []
df = polynomial_features(Xval, 3)
df_n = normalize_feature(df)
Xval = np.concatenate([np.ones((Xval.shape[0], 1)), np.array(df_n)], axis=1)
_lambda = 100
for i in range(1, m + 1):
    final_theta, _ = batch_gradient_decent(theta, X[: i,:], y[: i,:], epoch, _lambda=_lambda)
    train_cost.append(cost(final_theta, X[: i,:], y[: i,:]))
    cv_cost.append(cost(final_theta, Xval, yval))

train_cost, cv_cost = np.array(train_cost).reshape(len(train_cost)), np.array(cv_cost).reshape(len(cv_cost))

plt.figure()
plt.plot(np.arange(1, X.shape[0] + 1), train_cost, label='train cost')
plt.plot(np.arange(1, X.shape[0] + 1), cv_cost, label='cv cost')
plt.legend(loc='best')

#3.3 Selecting lambda using a cross validation set
l_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
train_cost, cv_cost = [], []
epoch = 5000

for l in l_list:
    final_theta, _ = batch_gradient_decent(theta, X, y, epoch, _lambda=l)
    train_cost.append(cost(final_theta, X, y))
    cv_cost.append(cost(final_theta, Xval, yval))

train_cost, cv_cost = np.array(train_cost).reshape(len(train_cost)), np.array(cv_cost).reshape(len(cv_cost))

plt.figure()
plt.plot(l_list, train_cost, label='train cost')
plt.plot(l_list, cv_cost, label='cv cost')
plt.legend(loc='best')

best_lambda = l_list[np.argmin(cv_cost)]
print("best lambda = ", best_lambda)

#3.4 Computing test set error
Xtest, ytest = np.array(data['Xtest']), np.array(data['ytest'])
df = polynomial_features(Xtest, 3)
df_n = normalize_feature(df)
Xtest = np.concatenate([np.ones((Xtest.shape[0], 1)), np.array(df_n)], axis=1)
epoch = 5000
alpha = 0.01
final_theta, _ = batch_gradient_decent(theta, X, y, epoch, alpha, _lambda=best_lambda)
test_error = cost(final_theta, Xtest, ytest)
print("test error = ", test_error)
plt.show()

