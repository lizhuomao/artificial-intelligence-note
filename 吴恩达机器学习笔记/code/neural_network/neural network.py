from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

#输出一张图片
def plot_image(image):
    #plt.figure()
    plt.matshow(image.reshape((20, 20)))
    plt.xlabel(())
    plt.ylabel(())

def plot_images(image):


#1.1 Dataset
data = loadmat('ex4data1.mat')
print(data['X'].shape, data['y'].shape)

#1.2 Visualizing the data
plot_image(data['X'][1024,:])
print(data['y'][1024])
plt.show()