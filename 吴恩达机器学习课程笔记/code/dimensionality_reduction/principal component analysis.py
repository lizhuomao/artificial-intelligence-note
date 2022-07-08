import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

def normalize(X):
    X_copy = X.copy()
    for i in range(X.shape[1]):
        X_copy[:, i] = (X_copy[:, i] - X_copy[:, i].mean()) / X_copy[:, i].std()
    return X_copy

def convariance_matrix(X):
    return X.T.dot(X) / X.shape[0]

def project_X(X, U, k):
    return X.dot(U[:, :k])

def pca(X):
    X_norm = normalize(X)
    sigma = convariance_matrix(X_norm)
    U, S, V = np.linalg.svd(sigma)
    return U, S, V

def recover_data(Z, U):
    return Z.dot(U[:, :Z.shape[1]].T)

#2 Principal Component Analysis

#2.1 Example Dataset

data = sio.loadmat('ex7data1.mat')

df = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
print(df.shape)

plt.scatter(df['X1'], df['X2'])



#2.2 Implementing PCA

#2.3 Dimensionality Reduction with PCA

X = np.array(df)
X_norm = normalize(X)

sigma = convariance_matrix(X_norm)
print(sigma)
Z = project_X(X, sigma, 1)
print(Z[:10])

plt.figure()
plt.scatter(Z, np.zeros(Z.shape), s=5)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.xaxis.set_ticks_position(('bottom'))
ax.spines['bottom'].set_position(('data', 0))

#2.3.1 Projecting the data onto the principal components

U, S, V = pca(X)
print(U)
Z = project_X(X, U, 1)

plt.figure()
plt.scatter(Z, np.zeros(Z.shape), s=5)
ax = plt.gca()

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.xaxis.set_ticks_position(('bottom'))
ax.spines['bottom'].set_position(('data', 0))
ax.set_yticks([])

plt.show()

#visualizing the projections
Z = project_X(X_norm, U, 1)
X_recover = recover_data(Z, U)

plt.scatter(X_norm[:, 0], X_norm[:,1], s=5, c='blue')
plt.scatter(X_recover[:, 0], X_recover[:, 1], s=5, c='green')

plt.show()

#2.4 Face Image Dataset
faces = sio.loadmat('ex7faces.mat')

X = np.array(faces.get('X'))
print(X.shape)
plt.imshow(X[0].reshape((32, 32)).T)


_, ax = plt.subplots(10, 10)

for i in range(10):
    for j in range(10):
        ax[i, j].imshow(X[i * 10 + j].reshape(32, 32).T, cmap='binary')
        ax[i, j].set_axis_off()

#2.4.1 PCA on Faces
U, _, _ = pca(X)
print(U.shape)

Z = project_X(X, U, k=100)

_, ax = plt.subplots(8, 8)

for i in range(8):
    for j in range(8):
        ax[i, j].imshow(Z[i * 8 + j].reshape(10, 10).T, cmap='binary')
        ax[i, j].set_axis_off()

X_recover = recover_data(Z, U)

_, ax = plt.subplots(8, 8)

for i in range(8):
    for j in range(8):
        ax[i, j].imshow(X_recover[i * 8 + j].reshape(32, 32).T, cmap='binary')
        ax[i, j].set_axis_off()

plt.show()