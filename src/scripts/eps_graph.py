import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: eps_graph.py in.txt")
        exit(1)
    file = sys.argv[1]
    data = np.loadtxt(file)
    plt.plot(data[0], data[1])
    plt.xlabel('Epsilon')
    plt.ylabel('Time (microseconds)')
    plt.show()