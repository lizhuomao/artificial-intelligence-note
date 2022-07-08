import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_X(data):
    #构造特征 传入原始数据 返回构造好的自变量
    ones = pd.DataFrame({'ones': np.ones(len(data))})
    return np.array(pd.concat([ones, data.iloc[:, :-1]], axis=1)) #合并数据 返回ndarray
def get_Y(data):
    #构造标签
    return np.array(data.iloc[:, 1]).reshape(data.shape[0],1)
def linear_regression_cost(theta, X, Y):
    #theta NX1矩阵 线性回归参数 N:特征数
    #X MXN矩阵 特征 M:样本数 N:特征数
    #Y MX1矩阵 标签

    #代价函数 cost function
    cost_function = ((X.dot(theta) - Y).T).dot(X.dot(theta) - Y) / (2 * X.shape[0]) #MXN * NX1 = MX1, 1XM * MX1 = 1X1
    return cost_function

def gradient(theta, X, Y):
    #代价函数求导即为代价函数的gradient
    return (X.T).dot((X.dot(theta) - Y)) / X.shape[0] #2XN * NX1(NX2 * 2X1 - NX1)

def batch_gradient_decent(theta, X, Y, epoch, alpha=0.01):
    #epoch: 批处理次数
    #alpha: 学习率
    #返回训练好的参数 代价函数优化过程
    cost_data = [linear_regression_cost(theta, X, Y)]
    _theta = theta.copy() #权重
    for i in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, Y) #更新参数
        cost_data.append(linear_regression_cost(_theta, X, Y)) #记录优化过程
    return _theta, np.array(cost_data).reshape(len(cost_data),1)

def normalization_feature(data):
    #将特征得标准差控制在1，均值控制在0
    for i in range(data.shape[1]):
        data.iloc[:, i] = (data.iloc[:, i] - data.iloc[:, i].mean()) / data.iloc[:, i].std()
    return data

def normal_equation(X, Y):
    #原理见笔记
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)
    return theta

data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])

#linear regression with one variables
#batch gradient decent
X = get_X(data)
Y = get_Y(data)
theta = np.zeros((X.shape[1], 1))
print(linear_regression_cost(theta, X, Y))
epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, Y, epoch)
print(cost_data[:5, 0])
print(final_theta)

plt.figure()
plt.plot(np.arange(0, epoch + 1).reshape(epoch + 1, 1), cost_data)
plt.xlabel('epoch')
plt.ylabel('cost')

plt.figure()
b = final_theta[0]
m = final_theta[1]
plt.scatter(data['population'], data['profit'], s=75, c='#66ccff' , alpha=0.5, edgecolors='#9999ff')
plt.plot(data.iloc[0], data.iloc[0] * m + b)

#linear regression with multiple variables
raw_data = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])
print(raw_data.head())

#feature Normalization
data = normalization_feature(raw_data)
print(data.head())
X = get_X(data)
print(X.shape)
Y = get_Y(data)
print(Y.shape)
theta = np.zeros((X.shape[1], 1))

#multi-var batch gradient decent(多变量批量梯度下降)
alpha = 0.01
epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, Y, epoch, alpha)

plt.figure()
plt.plot(np.arange(0, epoch + 1).reshape(epoch + 1, 1),cost_data)
plt.xlabel('epoch')
plt.ylabel('cost')
print(final_theta)

#learning rate
alphas = np.logspace(-1, -5, num=4)
alphas = np.sort(np.concatenate((alphas, alphas * 3)))
print(alphas)
epoch = 50
plt.figure()
for alpha in alphas:
    _, cost_data = batch_gradient_decent(theta, X, Y, epoch, alpha)
    plt.plot(np.arange(0, epoch + 1).reshape(epoch + 1, 1), cost_data, label=alpha)
plt.legend(loc='best')

#normal equation
final_theta = normal_equation(X, Y)
print(final_theta)

plt.show()