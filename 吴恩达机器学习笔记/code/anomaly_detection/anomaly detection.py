import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

def multivariate_gaussian(X, mu, sigma, n):
    y = []
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    print(sigma_inv, sigma_det)
    i = 0
    for x in X:
        y.append(np.exp(-0.5 * (x - mu).dot(sigma_inv).dot(x - mu)) / (np.power(2 * np.pi, n / 2) * np.sqrt(sigma_det)))

    return np.array(y)

def F1_score(y, y_pred):
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(y.shape[0]):
        if y[i] == y_pred[i] and y[i] == 1:
            tp += 1
        if y[i] == y_pred[i] and y[i] == 0:
            tn += 1
        if y[i] != y_pred[i] and y[i] == 1:
            fn += 1
        if y[i] != y_pred[i] and y[i] == 0:
            fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 * precision * recall / (precision + recall)

#1 Anomaly detection

data = sio.loadmat('ex8data1.mat')
print(data.keys())
df = pd.DataFrame(data.get('X'), columns=['Latency', 'Throughput'])

plt.scatter(df['Latency'], df['Throughput'], s=5)
plt.xlabel('Latency')
plt.ylabel('Throughput')
plt.show()

#1.1 Gaussian distribution

X = np.array(df)
print(X.shape)
mu = X.mean(axis=0)
print(mu)
cov = np.cov(X.T)
print(cov)

#1.2 Estimating parameters for a Gaussian

grid_x = np.meshgrid(np.linspace(df['Latency'].min() - 0.5, df['Latency'].max() + 0.5, 100), np.linspace(df['Throughput'].min() - 0.5, df['Throughput'].max() + 0.5, 100))
flat_x = np.column_stack((grid_x[0].ravel(), grid_x[1].ravel()))
flat_y = multivariate_gaussian(flat_x, mu, cov, X.shape[0])
grid_y = flat_y.reshape(grid_x[0].shape)
print(grid_x[0].shape, grid_y.shape)

plt.contourf(grid_x[0], grid_x[1], grid_y, cmap='spring')
plt.scatter(df['Latency'], df['Throughput'], s=5)

plt.show()

#1.3 Selecting the threshold, epsilon
Xval = np.array(data.get('Xval'))
yval = np.array(data.get('yval'))
pval = multivariate_gaussian(Xval, mu, cov, X.shape[0])
epsilon = np.linspace(pval.min(), pval.max(), 1000)

f1_score = []
for e in epsilon:
    y_pred = (pval <= e).astype('int')
    f1_score.append(F1_score(yval, y_pred))

max_fs_idx = np.argmax(f1_score)
best_e = epsilon[max_fs_idx]
print("Best epsilon:{}, Best F1-score:{}".format(best_e, f1_score[max_fs_idx]))

grid_x = np.meshgrid(np.linspace(df['Latency'].min() - 0.5, df['Latency'].max() + 0.5, 100), np.linspace(df['Throughput'].min() - 0.5, df['Throughput'].max() + 0.5, 100))
flat_x = np.column_stack((grid_x[0].ravel(), grid_x[1].ravel()))
flat_y = multivariate_gaussian(flat_x, mu, cov, X.shape[0])
grid_y = flat_y.reshape(grid_x[0].shape)

y = multivariate_gaussian(X, mu, cov, X.shape[0])

plt.contourf(grid_x[0], grid_x[1], grid_y, cmap='spring')
plt.scatter(df['Latency'], df['Throughput'], s=5, c=(y <= best_e))
plt.show()