from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

#输出一张图片
def plot_image(image):
    #plt.figure()
    plt.matshow(image.reshape((20, 20)))
    plt.xticks(())
    plt.yticks(())

def plot_images(image):
    f, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    for i in range(10):
        for j in range(10):
            ax[i, j].matshow(image['X'][np.random.randint(0, 5000)].reshape((20, 20)), cmap=matplotlib.cm.binary)
            plt.xticks(())
            plt.yticks(())

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
        #print('epoch = {}'.format(i))
        _theta = _theta - alpha * gradient(X, Y, _theta, _lambda)
        cost_data.append(cost(X, Y, _theta, _lambda))
    return _theta, np.array(cost_data).reshape(len(cost_data), 1)
#预测
def predit(X, theta):
    return (hypothesis(X, theta) >= 0.5).astype(int)

#1.1 Dataset
data = loadmat('ex4data1.mat')
print(data['X'].shape, data['y'].shape)

#1.2 Visualizing the data
#plot_image(data['X'][1024,:])
#print(data['y'][1024])
plot_images(data)

#1.3 Vectorizing Logistic Regression
X = np.concatenate([np.ones(data['X'].shape[0]).reshape(data['X'].shape[0], 1), np.array(data['X'])], axis=1)
y_vector = []

for i in range(10):
    y_vector.append((data['y'] % 10 == i).astype(int))
y = np.array(y_vector).reshape(10, 5000)
print(X.shape, y.shape)
print(y)

#1.4 One-vs-all Classification
theta = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
epoch = 5000
alpha = 0.01
'''
final_theta, cost_data = batch_gradient_decent(X, y[0].reshape(5000, 1), theta, epoch, alpha, 1)
plt.figure()
plt.plot(np.arange(epoch + 1).reshape(epoch + 1, 1), cost_data)
print(np.mean(predit(X, final_theta) == y[0].reshape(5000, 1)))
'''
plt.figure()
_, ax = plt.subplots(3, 4, sharex=True, sharey=True)
final_thetas = np.array([])
for i in range(10):
    print("training number {}".format((i)))
    final_theta, cost_data = batch_gradient_decent(X, y[i].reshape(5000, 1), theta, epoch, alpha, 1)
    ax[int(i / 4), int(i % 4)].plot(np.arange(epoch + 1).reshape(epoch + 1, 1), cost_data)
    print(np.mean(predit(X, final_theta) == y[i].reshape(5000, 1)))
    np.append(final_thetas, final_theta)
print(final_thetas.shape)
plt.show()