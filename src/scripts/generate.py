
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #data = np.random.uniform(size=(100000, 4))
    data = np.random.normal(size=(100000, 4))
    #plt.scatter(data[0], data[1])
    #plt.show()
    #np.transpose()
    np.savetxt('data/normal_4d_e5.txt', data, fmt='%.10f')
    #print(data[0])