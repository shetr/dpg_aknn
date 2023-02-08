import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: display.py in.txt")
        exit(1)
    file = sys.argv[1]
    data = np.loadtxt(file)
    data = np.transpose(data)
    plt.scatter(data[0], data[1])
    plt.show()
    #print(data[0])