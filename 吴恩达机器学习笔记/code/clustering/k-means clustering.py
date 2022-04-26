import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def random_init(df, k):
    return np.array(df.sample(k))

def find_cluster(x, centroids):
    distance = []
    for centroid in centroids:
        distance.append(np.sqrt(np.power(x - centroid, 2).sum()))
    distance = np.array(distance)
    return np.argmin(distance)

def assign_cluster(df, centroids):
    C = []
    for i in range(df.shape[0]):
        C.append(find_cluster(df.iloc[i], centroids))
    return np.array(C)

def new_centroids(df, C):
    df_C = df.copy()
    df_C['C'] = C
    return np.array(df_C.groupby('C', as_index=False).mean().sort_values(by='C').drop('C', axis=1))

def cost(df, centroids, C):
    m = df.shape[0]

    distances = 0
    for i in range(m):
        distances += np.sqrt(np.power(df.iloc[i] - centroids[C[i]], 2).sum())
    return distances / m


def k_means_part(df, k, epoch=100, tol=0.00001):
    centroids = random_init(df, k)
    cost_data = []

    for i in range(epoch):
        print('epoch = ', i)
        C = assign_cluster(df, centroids)
        centroids = new_centroids(df, C)
        cost_data.append(cost(df, centroids, C))

        if len(cost_data) > 1:
            if (np.abs(cost_data[-1] - cost_data[-2])) / cost_data[-1] < tol:
                break

    return C, centroids, cost_data[-1]

def k_means(df, k, epoch=100, n_init=10):
    result = []
    for i in range(n_init):
        result.append(k_means_part(df, k, epoch))
    result = np.array(result)
    least_cost_idx = np.argmin(result[:, -1])
    return result[least_cost_idx]

#1 K-means Clustering

data = sio.loadmat('ex7data2.mat')
df = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])

plt.scatter(df['X1'], df['X2'], s=10)
plt.xlabel('X1')
plt.ylabel('X2')

plt.show()

#1.1 Implementing K-means

#1.1.1 Finding closest centroids
init_centroids = random_init(df, 3)
print(init_centroids)
x = np.array([1, 1])
print(find_cluster(x, init_centroids))
C = assign_cluster(df, init_centroids)

plt.scatter(df['X1'], df['X2'], s=10, c=C)
plt.xlabel('X1')
plt.ylabel('X2')

plt.show()

#1.1.2 Computing centroid means
print(new_centroids(df, C))

#1.2 K-means on example dataset
#1.3 Random initialization
final_C, final_centroid, cost_data = k_means_part(df, 3)
print(cost_data)

plt.scatter(df['X1'], df['X2'], s=10, c=final_C)
plt.xlabel('X1')
plt.ylabel('X2')



best_C, best_centroids, least_cost = k_means(df, 3, n_init=1)
print('least_cost = ',least_cost)

plt.figure()
plt.scatter(df['X1'], df['X2'], s=10, c=best_C)
plt.title('best C')
plt.xlabel('X1')
plt.ylabel('X2')

plt.show()

#1.4 Image compression with K-means

# cast to float, you need to do this otherwise the color would be weird after clustring
image = plt.imread('bird_small.png') / 255.
plt.imshow(image)
print(image.shape)
df = pd.DataFrame(image.reshape((128 * 128 , 3)))

C, centroids, cost = k_means(df, 16, epoch=10,n_init=3)
comppressed_image = centroids[C].reshape((128, 128, 3))
plt.imshow(comppressed_image)

plt.show()
