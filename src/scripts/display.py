
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt('data/normal_2d_e3.txt')
    data = np.transpose(data)
    plt.scatter(data[0], data[1])
    plt.show()
    #print(data[0])