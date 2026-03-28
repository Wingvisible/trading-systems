import matplotlib.pyplot as plt
import numpy as np
import k_means 
def plot(datapoints: list, X:np.array, centroids):
    
    labels = k_means.assign_closest_centroid(X, centroids)
    K = centroids.shape[0]
    colors = ['red', 'blue', 'green']
    for i in range(K):
        index_list = np.where(labels == i)[0] 
        x_list = index_list
        y_list = datapoints[index_list]
        plt.scatter(x_list, y_list, label = f"cluster {i}", s=1)
    
    plt.legend()
    plt.show()

    