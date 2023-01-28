
import numpy as np
import matplotlib.pyplot as plt

def gen_clusters(size, dim, clusters_count, sigma, max_val=1):
    cluster_size = size // clusters_count
    scale = np.full(shape=dim, fill_value=sigma)
    locs = np.random.uniform(high=max_val, size=(clusters_count, dim))
    cluster_data = [np.random.normal(loc=locs[i], scale=scale, size=(cluster_size, dim)) for i in range(clusters_count)]
    data = np.vstack(cluster_data)
    return data

if __name__ == "__main__":
    #data = np.random.uniform(size=(100000, 4))
    #data = np.random.normal(size=(100000, 4))
    data = gen_clusters(10000, 2, 10, 0.05)
    
    #np.savetxt('data/normal_4d_e5.txt', data, fmt='%.10f')
    #print(data[0])
    
    data = np.transpose(data)
    plt.scatter(data[0], data[1])
    plt.show()