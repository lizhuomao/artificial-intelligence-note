import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

#1 Support Vector Machines

#1.1 Example Data 1
data = sio.loadmat('ex6data1.mat')
print(data.keys())
df = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
df['y'] = data.get('y')
print(df.head())

plt.scatter(df['X1'], df['X2'], c=df['y'])

# C = 1
svc = svm.SVC(C=1, kernel='linear')
svc.fit(df[['X1', 'X2']], df['y'])
print("C = 1:",svc.score(df[['X1', 'X2']], df['y']))

grid_x = np.meshgrid(np.linspace(df['X1'].min() - 1, df['X1'].max() + 1, 500), np.linspace(df['X2'].min() - 1, df['X2'].max() + 1, 500))
flat_x = np.column_stack((grid_x[0].ravel(), grid_x[1].ravel()))
flat_y = svc.predict(flat_x)
grid_y = flat_y.reshape(grid_x[0].shape)
plt.figure()
plt.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='gray')
plt.scatter(df['X1'], df['X2'], c=df['y'])
df['SVM1 confidence'] = svc.decision_function(df[['X1', 'X2']])

svc = svm.SVC(C=100, kernel='linear')
svc.fit(df[['X1', 'X2']], df['y'])
print("C = 100:",svc.score(df[['X1', 'X2']], df['y']))

flat_y = svc.predict(flat_x)
grid_y = flat_y.reshape(grid_x[0].shape)
plt.figure()
plt.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='gray')
plt.scatter(df['X1'], df['X2'], c=df['y'])
df['SVM100 confidence'] = svc.decision_function(df[['X1', 'X2']])

print(df.head())

plt.show()

#1.2 SVM with Gaussian Kernels

#1.2.1 Gaussian Kernel
def gaussian_kernel(x, l, sigma):
    return np.exp(-np.power(x - l, 2).sum() / (2 * (sigma ** 2)))

x = np.array([1, 2, 1])
l = np.array([0, 4, -1])
sigma = 2

print(gaussian_kernel(x, l, sigma))

#1.2.2 Example Dataset 2
data = sio.loadmat('ex6data2.mat')
df = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
df['y'] = data.get('y')
print(df.head())
plt.scatter(df['X1'], df['X2'], c=df['y'], s=5)

svc = svm.SVC(C=100, kernel='rbf', gamma=10)
svc.fit(df[['X1', 'X2']], df['y'])
print("example2: ", svc.score(df[['X1', 'X2']], df['y']))
grid_x = np.meshgrid(np.linspace(df['X1'].min() - 0.05, df['X1'].max() + 0.05, 500), np.linspace(df['X2'].min() - 0.05, df['X2'].max() + 0.05, 500))
flat_x = np.column_stack((grid_x[0].ravel(), grid_x[1].ravel()))
flat_y = svc.predict(flat_x)
grid_y = flat_y.reshape(grid_x[0].shape)

plt.figure()
plt.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='winter')
plt.scatter(df['X1'], df['X2'], c=df['y'], s=5)

plt.show()

#1.2.3 Example Dataset 2
data = sio.loadmat('ex6data3.mat')
train = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
train['y'] = data.get('y')

cv = pd.DataFrame(data.get('Xval'), columns=['X1', 'X2'])
cv['y'] = data.get('yval')

print(train.shape, cv.shape)
plt.scatter(train['X1'], train['X2'], c=train['y'], s=5)

c_l = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
sigma_l = c_l.copy()

c_s_l = np.meshgrid(c_l, sigma_l)
c_s_l = np.column_stack((c_s_l[0].ravel(), c_s_l[1].ravel()))

search = []

for c, s in c_s_l:
    svc = svm.SVC(C=c, gamma=s, kernel='rbf')
    svc.fit(train[['X1', 'X2']], train['y'])
    search.append(svc.score(cv[['X1', 'X2']], cv['y']))
best_score = search[np.argmax(search)]
best_param = c_s_l[np.argmax(search)]
print(best_score, best_param)

best_svc = svm.SVC(kernel='rbf', C=100, gamma=0.3)
best_svc.fit(train[['X1', 'X2']], train['y'])

grid_x = np.meshgrid(np.linspace(train['X1'].min() - 0.05, train['X1'].max() + 0.05, 1000), np.linspace(train['X2'].min() - 0.05, train['X2'].max() + 0.05, 1000))
flat_x = np.column_stack((grid_x[0].ravel(), grid_x[1].ravel()))
flat_y = best_svc.predict(flat_x)
grid_y = flat_y.reshape(grid_x[0].shape)

plt.figure()
plt.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='winter')
plt.scatter(train['X1'], train['X2'], c=train['y'], s=5)

plt.show()