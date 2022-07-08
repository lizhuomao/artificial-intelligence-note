import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

def cost(theta, X, Y, R):
    return np.power(np.multiply(X.dot(theta.T) - Y, R), 2).sum() / 2

def gradient(theta, X, Y, R):
    core = np.multiply(X.dot(theta.T) - Y, R)
    X_grad = core.dot(theta)
    theta_grad = core.T.dot(X)
    return theta_grad, X_grad

def regularized_cost(theta, X, Y, R, _lambda=1):
    return cost(theta, X, Y, R) + (_lambda / 2) * (X.dot(X.T).sum() + theta.dot(theta.T).sum())

def regularized_gradient(theta, X, Y, R, _lambda=1):
    theta_grad, X_grad = gradient(theta, X, Y, R)
    return theta_grad + _lambda * theta, X_grad + _lambda * X

def batch_gradient_decent(theta, X, Y, R, epoch, alpha=0.01, _lambda=1):
    cost_data = []
    _theta = theta.copy() #权重
    _X = X.copy()
    for i in range(epoch):
        print(i)
        theta_grad, X_grad = regularized_gradient(_theta, _X, Y, R, _lambda)
        _theta = _theta - alpha * theta_grad
        _X = _X - alpha * X_grad
        cost_data.append(regularized_cost(_theta, _X, Y, R, _lambda)) #记录优化过程
    return _theta, _X, np.array(cost_data)

#2 Recommender Systems

#2.1 Movie ratings dataset

data_Y_R = sio.loadmat('ex8_movies.mat')
print(data_Y_R.keys())
Y, R = np.array(data_Y_R.get('Y')), np.array(data_Y_R.get('R'))
print(Y.shape, R.shape)

data_param = sio.loadmat('ex8_movieParams.mat')
print(data_param.keys())
theta, X = np.array(data_param.get('Theta')), np.array(data_param.get('X'))
print(theta.shape, X.shape)

#2.2 Collaborative filtering learning algorithm

#2.2.1 Collaborative filtering cost function
print(cost(theta, X, Y, R))

#2.2.2 Collaborative filtering gradient
theta_grad, X_grad = gradient(theta, X, Y, R)
print(theta_grad.shape, X_grad.shape)

#2.2.3 Regularized cost function
print(regularized_cost(theta, X, Y, R))

#2.2.4 Regularized gradient
theta_grad, X_grad = regularized_gradient(theta, X, Y, R)
print(theta_grad.shape, X_grad.shape)

#2.3 Learning movie recommendations
movie_list = []

with open('movie_ids.txt', encoding='gbk') as f:
    for line in f:
        tokens = line.strip()
        movie_list.append(line.strip())

movie_list = np.array(movie_list)
print(movie_list[:10])

ratings = np.zeros(1682)

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

Y = np.insert(Y, 0, ratings, axis=1)
R = np.insert(R, 0, ratings != 0, axis=1)
print(Y.shape, R.shape)

features_n = 50
X = np.random.standard_normal((Y.shape[0], features_n))
theta = np.random.standard_normal((Y.shape[1], features_n))
print(X.shape, theta.shape)
Y_norm = Y - Y.mean()
print(Y_norm.mean())

epoch = 100
alpha = 0.0001
_lambda = 1000
final_theta, final_X, cost_data = batch_gradient_decent(theta, X, Y_norm, R, epoch, alpha, _lambda)

print(cost_data[:10])
print(cost_data[-10:])
plt.plot(np.arange(len(cost_data)), cost_data)


prediction = final_X.dot(final_theta.T)

my_preds = prediction[:, 0] + Y.mean()
idx = np.argsort(my_preds)[::-1]
print(my_preds[idx][:10])
print(idx.shape)
for m in movie_list[idx][:10]:
    print(m)
plt.show()