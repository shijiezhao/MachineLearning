import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import numpy as np


def plot(predictor, X, y, grid_size, filename):
    x_min, x_max = -6,6 #X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = -6,6 #X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.project(point))

    Z = np.array(result).reshape(xx.shape)
	# Plot decision boundary
    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.00001, 0.00001],
                 extend='both',
                 alpha=0.8)
    # Plot the X,y points...
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    # Plot Margin
    pl.contour(xx, yy, Z+1, [0.0], colors='b', linewidths=1, origin='lower')
    pl.contour(xx, yy, Z-1, [0.0], colors='b', linewidths=1, origin='lower')
    
    # Plot S.V.s on margin
    SV = predictor._margin_support_vectors
    print SV
    print SV.shape
    pl.scatter(SV[:,0], SV[:,1], s=100, c="k")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(filename)
    plt.show()