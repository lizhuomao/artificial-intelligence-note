import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#公式的推导见markdown

#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#hypothesis function
def hypothesis(X, theta):
   return sigmoid(X.dot(theta))

#regularized cost function
def cost(X, Y, theta, _lambda):
    c_function = -Y.T.dot(np.log(hypothesis(X, theta))) - (1 - Y.T).dot(np.log(1 - hypothesis(X, theta)))
    c_function += (_lambda / (2 * X.shape[0])) * theta[1:].T.dot(theta[1:])
    return c_function

#regularized gradient
def gradient(X, Y, theta, _lambda):
    l_term = np.concatenate([np.array([[0]]), (_lambda / X.shape[0]) * theta[1:]])
    return (X.T.dot(hypothesis(X, theta) - Y) / X.shape[0]) + l_term

#logistic regression
def batch_gradient_decent(X, Y, theta, epoch, alpha, _lambda):
    cost_data = [cost(X, Y, theta, _lambda)]
    _theta = theta.copy()
    for i in range(epoch):
        _theta = _theta - alpha * gradient(X, Y, _theta, _lambda)
        cost_data.append(cost(X, Y, _theta, _lambda))
    return _theta, np.array(cost_data).reshape(len(cost_data), 1)

#获取特征
def get_X(data):
    ones = pd.DataFrame({'ones' : np.ones(data.shape[0])})
    return np.array(pd.concat([ones, data.iloc[:, : -1]], axis=1))

#获取标签
def get_Y(data):
    return np.array(data.iloc[:, -1]).reshape(data.shape[0], 1)

#特征映射
def feature_mapping(x1, x2, power):
    data = {}
    for i in np.arange(power + 1):
        for j in np.arange(i + 1):
            data["f{}{}".format(i - j, j)] = np.power(x1, i - j) * np.power(x2, j)
    return pd.DataFrame(data)

#预测
def predit(X, theta):
    return (hypothesis(X, theta) >= 0.5)

data = pd.read_csv('ex2data2.txt', names=['Microchip test 1', 'Microchip test 2', 'result'])
print(data.head())
print(data.shape)
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data.iloc[:, 2])

#Feature mapping
#print(feature_mapping(data.iloc[:, 0], data.iloc[:, 1], 6).head())
x1 = data.iloc[:, 0]
x2 = data.iloc[:, 1]
X = np.array(feature_mapping(x1, x2, 6))
print(X.shape)
Y = get_Y(data)
print(Y.shape)
theta = np.zeros(X.shape[1]).reshape((X.shape[1], 1))
print(theta.shape)
#print(cost(X, Y, theta, 1))
#print(gradient(X, Y, theta, 1))

epoch = 5000
alpha = 0.01
#final_theta, cost_data = batch_gradient_decent(X, Y, theta, epoch, alpha, 0)
#final_theta, cost_data = batch_gradient_decent(X, Y, theta, epoch, alpha, 100)
final_theta, cost_data = batch_gradient_decent(X, Y, theta, epoch, alpha, 1)

#Plotting the decision boundary
test_data = {"x1":[], "x2":[]}
for i in np.linspace(-1, 1.5, 3000):
    for j in np.linspace(-1, 1.5, 3000):
        test_data["x1"].append(i)
        test_data["x2"].append(j)
test_df = pd.DataFrame(test_data)
print(test_df.shape)
test_map = feature_mapping(test_df.iloc[:, 0], test_df.iloc[:, 1], 6)
print(test_map.shape)
test_map = test_map[np.abs(np.array(test_map).dot(final_theta)) < 2 * 10 ** -3]
plt.scatter(test_map["f10"], test_map["f01"], c="#66ccff")

plt.figure()
plt.plot(np.arange(epoch + 1).reshape(epoch + 1, 1), cost_data)
plt.show()
