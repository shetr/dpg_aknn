import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: queue_graph.py in.txt")
        exit(1)
    file = sys.argv[1]
    data = np.loadtxt(file)
    plt.plot(data[0], data[1], label='linear')
    plt.plot(data[0], data[2], label='heap')
    plt.plot(data[0], data[3], label='std')
    plt.legend(title='Queue type:')
    plt.xlabel('K nearest neighbors')
    plt.ylabel('Time (microseconds)')
    plt.show()