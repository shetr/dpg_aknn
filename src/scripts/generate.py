
import numpy as np
import matplotlib.pyplot as plt

def gen_clusters(size, dim, clusters_count, sigma_min, sigma_max, max_val=1):
    cluster_size = size // clusters_count
    scales = np.random.uniform(low=sigma_min, high=sigma_max, size=(clusters_count))
    locs = np.random.uniform(high=max_val, size=(clusters_count, dim))
    cluster_data = [np.random.normal(loc=locs[i], scale=scales[i], size=(cluster_size, dim)) for i in range(clusters_count)]
    data = np.vstack(cluster_data)
    return data

def gen_clusters_custom(size, dim, clusters_count, scales, locs):
    cluster_size = size // clusters_count
    cluster_data = [np.random.normal(loc=locs[i], scale=scales[i], size=(cluster_size, dim)) for i in range(clusters_count)]
    data = np.vstack(cluster_data)
    return data

if __name__ == "__main__":
    #data = np.random.uniform(size=(100000, 4))
    #data = np.random.normal(size=(100000, 4))
    #data = gen_clusters(10000, 2, 5, 0.01, 0.1)
    data = gen_clusters_custom(10000000, 2, 5, np.array([0.25, 0.15, 0.1, 0.075, 0.05]), np.array([[0.3, 0.3], [0.8, 0.9], [1, 0], [0, 0.9], [-0.1, 0.5]]))
    #data = gen_clusters_custom(1000, 3, 6, np.array([0.25, 0.15, 0.1, 0.075, 0.06, 0.05]), np.array([[0.3, 0.3, 0.3], [0.8, 0.9, 0], [1, 0, 0], [0, 0.9, 1], [-0.1, 0.5, 0.9], [0.8, 0.9, 0.7]]))
    #data = gen_clusters_custom(1000, 4, 6, np.array([0.25, 0.15, 0.1, 0.075, 0.06, 0.05]), np.array([[0.3, 0.3, 0.3, 0.3], [0.8, 0.9, 0, 1], [1, 0, 0, 0.5], [0, 0.9, 1, 0], [-0.1, 0.5, 0.9, 0], [0.8, 0.9, 0.7, 0.8]]))
    
    np.savetxt('out/clusters_2d_e7.txt', data, fmt='%.10f')
    #print(data[0])
    
    #data = np.transpose(data)
    #plt.scatter(data[0], data[1])

    #ax = plt.axes(projection ="3d")
    #ax.scatter(data[0], data[1], data[2])

    #plt.show()