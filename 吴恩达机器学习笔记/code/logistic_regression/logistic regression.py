import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#公式的推导见markdown
#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#hypothesis function
def hypothesis(X, theta):
   z = X.dot(theta)
   return sigmoid(z)

#cost function
def cost(X, Y, theta):
    c_function = -Y.T.dot(np.log(hypothesis(X, theta))) - (1 - Y.T).dot(np.log(1 - hypothesis(X, theta)))
    return 1 / X.shape[0] * c_function

#gradient
def gradient(X, Y, theta):
    return (1 / X.shape[0]) * X.T.dot(hypothesis(X, theta) - Y)

#logistic regression
def logistic_regression(X, Y, theta, epoch, alpha=0.01):
    cost_data = [cost(X, Y, theta)]
    _theta = theta.copy()
    for i in range(epoch):
        _theta = -theta - gradient(X, Y, _theta)
        cost_data.append(cost(X, Y, theta))
    return _theta, np.array(cost_data).reshape(len(cost_data), 1)

#获取特征
def get_X(data):
    ones = pd.DataFrame({'ones' : np.ones(data.shape[0])})
    return np.array(pd.concat([ones, data.iloc[:, : -1]], axis=1))

#获取标签
def get_Y(data):
    return np.array(data.iloc[:, -1]).reshape(data.shape[0], 1)


data = pd.read_csv('ex2data1.txt', names=['exam1 score', 'exam2 score', 'admission'])
print(data.head())

#1.1 Visualizing the data
plt.figure()
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=data.iloc[:,2])
plt.xlabel('exam1 score')
plt.ylabel('exam2 score')

#1.2 Implementation
X = get_X(data)
Y = get_Y(data)

theta = np.zeros(data.shape[1]).reshape(data.shape[1], 1)
print(X.shape, Y.shape, theta.shape)

#sigmoid(0) = 0.5
print(hypothesis(X[1], theta))
#plt.figure()
#plt.plot(np.linspace((1, -1),))

#cost function and gradient
epoch=500
final_theta, cost_data = logistic_regression(X, Y, theta, epoch)
print(final_theta)
print(cost_data[:5])
plt.plot(data.iloc[0], (-final_theta[0, 0] - data.iloc[0] * final_theta[1, 0]) / final_theta[2,0])
plt.figure()
plt.plot(np.arange(epoch + 1).reshape(epoch + 1, 1), cost_data)

plt.show()