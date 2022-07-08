import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#hypothesis function
def hypothesis(X, theta):
   return sigmoid(X.dot(theta))

#feedforward Propagation function
def feed_forward(X, weight1, weight2):
    #print(weight1.shape, weight2.shape) #(25, 401) (10, 26)
    # layer1
    a1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1) #5000 401
    z2 = a1.dot(weight1.T)
    #print(z2.shape) #5000 25
    # layer2
    a2 = np.concatenate([np.ones((X.shape[0], 1)), sigmoid(z2)], axis=1) #5000 26
    z3 = a2.dot(weight2.T)
    #print(z3.shape) #5000 10
    # layer3
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def cost(weight1, weight2, X, Y):
    _, _, _, _, h = feed_forward(X, weight1, weight2)
    c_function = -np.multiply(Y, np.log(h)) - np.multiply((1 - Y), np.log(1 - h))
    return c_function.sum() / X.shape[0]

#regularization
def regularized_cost(weight1, weight2, X, Y, _lambda = 1):
    rest = (_lambda / (2 * X.shape[0])) * np.power(weight1[:, 1:], 2).sum()
    rest += (_lambda / (2 * X.shape[0])) * np.power(weight2[:, 1:], 2).sum()
    return cost(weight1, weight2, X, Y) + rest

#sigmoid gradient
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

#back propagation
def back_propagation(weight1, weight2, X, y):
    m = X.shape[0]

    #set delta = 0
    delta1 = np.zeros(weight1.shape)#(25, 401)
    delta2 = np.zeros(weight2.shape) #(10, 26)

    #forward propagation
    a1, z2, a2, z3, h = feed_forward(X, weight1, weight2)
    for i in range(m):
        a1_i = a1[i, :].reshape((1, 401)) #(1, 401)
        z2_i = z2[i, :].reshape((1, 25)) #(1, 25)
        a2_i = a2[i, :].reshape((1, 26)) #(1, 26)
        h_i = h[i, :].reshape((1, 10)) #(1, 10)
        y_i = y[i, :].reshape((1, 10)) #(1, 10)

        #Output layer
        d3_i = h_i - y_i #(1, 10)
        #Hidden layer
        z2_i = np.concatenate([np.ones((z2_i.shape[0], 1)), z2_i], axis=1) #(1, 26)
        d2_i = np.multiply(d3_i.dot(weight2), sigmoid_gradient(z2_i)) #(1, 26)
        d2_i = d2_i[0, 1:].reshape((1, 25))
        delta2 += d3_i.T.dot(a2_i)
        delta1 += d2_i.T.dot(a1_i)

    return delta1 / m, delta2 / m

def cost_ck(w_arr, X, Y):
    weight1 = w_arr[: 25 * 401].reshape((25, 401))
    weight2 = w_arr[25 * 401 :].reshape((10, 26))
    _, _, _, _, h = feed_forward(X, weight1, weight2)
    c_function = -np.multiply(Y, np.log(h)) - np.multiply((1 - Y), np.log(1 - h))
    return c_function.sum() / X.shape[0]

#Gradient checking
def gradient_checking(weights, X, y, epsilon=0.0001):
    w_arr = np.concatenate((np.ravel(np.array(weights['Theta1'])), np.ravel(np.array(weights['Theta2']))))
    w_arr = w_arr.reshape((1, w_arr.shape[0]))
    print(w_arr.shape)
    w_arr = np.ones((w_arr.shape[1], 1)).dot(w_arr)
    print(w_arr.shape)
    plus_arr = w_arr + (np.identity(w_arr.shape[0]) * epsilon)
    minus_arr = w_arr - (np.identity(w_arr.shape[0]) * epsilon)
    gradient_definition = []
    for i in range(w_arr.shape[0]):
        gradient_definition.append((cost_ck(plus_arr[i], X, y) - cost_ck(minus_arr[i], X, y)) / (2 * epsilon))
    weight1 = np.array(weights['Theta1'])
    weight2 = np.array(weights['Theta2'])
    w1, w2 = back_propagation(weight1, weight2, X, y)
    gradient_definition = np.array(gradient_definition)
    gradient_bp = np.concatenate((np.ravel(w1), np.ravel(w2)))
    print(gradient_bp.shape, gradient_definition.shape)
    # relative difference
    diff = np.linalg.norm(gradient_definition - gradient_bp) / np.linalg.norm(gradient_definition + gradient_bp)
    return diff

#gradient decent
def batch_gradient_decent(X, Y, weight1, weight2, epoch, alpha=0.01):
    cost_data = [cost(weight1, weight2, X, Y)]
    w1 = weight1.copy()
    w2 = weight2.copy()
    for i in range(epoch):
        #_theta = _theta - alpha * gradient(X, Y, _theta, _lambda)
        print("epoch = {}".format(i))
        _w1, _w2 = back_propagation(w1, w2, X, y)
        w1 = w1 - alpha * _w1
        w2 = w2 - alpha * _w2
        cost_data.append(cost(w1, w2, X, Y))
    return w1, w2, np.array(cost_data).reshape(len(cost_data), 1)

weights = sio.loadmat('ex3weights.mat')

data = sio.loadmat('ex4data1.mat')
X = np.array(data['X'])
y_vector = []

for i in range(10):
    y_vector.append((data['y'] % 10 == i).astype(int))
y = np.array(y_vector).reshape(10, 5000)
y = y.T
print(X.shape, y.shape)

#cost funtion
weight1 = np.array(weights['Theta1'])
weight2 = np.array(weights['Theta2'])
print(regularized_cost(weight1, weight2, X, y))

#back propagation
delta1, delta2 = back_propagation(weight1, weight2, X, y)
print(delta1.shape, delta2.shape)

#gradient checking
#print(gradient_checking(weights, X, y))

#random initialization
init_weight1 = np.random.uniform(-0.12, 0.12, (weight1.shape))
init_weight2 = np.random.uniform(-0.12, 0.12, (weight2.shape))

epoch = 3000

final_w1, final_w2, cost_data = batch_gradient_decent(X, y, init_weight1, init_weight2, epoch)

plt.plot(np.arange(epoch + 1).reshape(epoch + 1, 1), cost_data)
plt.show()