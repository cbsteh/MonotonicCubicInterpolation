import numpy as np
from monospline import MonoSpline
import matplotlib.pyplot as plt

# xy-input.txt file contains pairs of (x, y) data -- no missing values.
# x-input.txt file contains a list of x values for y interpolation.
# results written in xy-output.txt file (complete data, including
#    original values in xy-input.txt file and interpolated (x, y) values.
if __name__ == "__main__":
    data = np.loadtxt('xy-input.txt', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    ms = MonoSpline(x, y)
    xi = np.loadtxt('x-input.txt')
    yi = ms.evaluate(xi)
    xyi = [[z[0], z[1]] for z in zip(xi, yi)]
    data = np.append(data, xyi, axis=0)
    data = data[data[:,0].argsort()]

    with open('xy-output.txt', 'w') as fout:
        for xy in data:
            fout.write(f'{xy[0]:.0f},{xy[1]:.2f}\n')

    plt.plot(x, y, '-', label='data')
    plt.scatter(xi, yi, c='r', marker='o', label='interpolated points')
    plt.legend(loc='lower right')
    plt.show()
