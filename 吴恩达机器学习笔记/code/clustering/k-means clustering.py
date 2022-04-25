import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def random_init(df, k):
    return np.array(df.sample(k))

def find_cluster(x, centroids, m):
    distance = np.power(x - centroids, 2).sum()
    return np.argmin(distance)

def assign_cluster(df, centroids):
    C = []
    for i in range(centroids.shape[0]):
        C.append(find_cluster(df.iloc[i], centroids))
    return np.array(C)

def new_centroids(df, C):
    df_C = df.copy()
    df_C['C'] = C



def k_means_part(df, k, epoch=100, tol=0.0001):
    centroids = random_init(df, k)
    cost_data = []

    for i in range(epoch):
        print('epoch = ', i)
        C = assign_cluster(df, centroids)
        centroids = new_centroids(df, C)

def k_means(df, k, epoch=100, n_init=10):
    result = []
    for i in range(n_init):
        result.append([k_means_part(df, k, epoch)])
    result = np.array(result)
    least_cost_idx = np.argmin(result[:, 1])
    return result[least_cost_idx]

#1 K-means Clustering

data = sio.loadmat('ex7data2.mat')
df = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])

plt.scatter(df['X1'], df['X2'], s=10)
plt.xlabel('X1')
plt.ylabel('X2')

plt.show()

#1.1 Implementing K-means

print(random_init(df, 3))


