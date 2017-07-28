from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Convert X and y to ndarray
    if (type(X) == pd.DataFrame):
        X = X.values
    if (type(y) == pd.DataFrame):
        y = y.value
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # y == cl returns an array of booleans that match class cl which are then used in X
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # Highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c='', edgecolors='k', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')



def plot_decision_regions3d(X_train, y_train, classifier, resolution=2.0):
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    markers = ('x', 'o', 'h', 'd', '^')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = X_train.ix[:,0].values
    y = X_train.ix[:,1].values
    z = X_train.ix[:,2].values
    unique_vals = np.unique(y_train)
    cmap = ListedColormap(colors[:len(unique_vals)])
    for idx, cl in enumerate(np.unique(y_train)):
        ax.scatter(x[y_train == idx], y[y_train == idx], z[y_train == idx], c=cmap(idx), marker=markers[idx])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # plot the decision surface
    x_min, x_max = x.min()-1, x.max()+1
    y_min, y_max = y.min()-1, y.max()+1
    z_min, z_max = z.min()-1, z.max()+1

    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution), np.arange(z_min, z_max, resolution))
    val = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).T
    Z = classifier.predict(val)
    Z = Z.reshape(xx.shape)

    for idx, cl in enumerate(np.unique(unique_vals)):
        ax.scatter(xx[Z == idx], yy[Z == idx], zz[Z == idx], c=cmap(idx), marker='.')

    plt.show()

# Only works with Logistic Regression (or Linear SVC with a bit of tweaking) right now
# See https://stackoverflow.com/questions/36232334/plotting-3d-decision-boundary-from-linear-svm
def plot_decision_plane(X_train, y_train, classifier, resolution=0.1):
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    markers = ('x', 'o', 'h', 'd', '^')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = X_train.ix[:,0].values
    y = X_train.ix[:,1].values
    z = X_train.ix[:,2].values
    unique_vals = np.unique(y_train)
    cmap = ListedColormap(colors[:len(unique_vals)])
    for idx, cl in enumerate(np.unique(y_train)):
        ax.scatter(x[y_train == idx], y[y_train == idx], z[y_train == idx], c=cmap(idx), marker=markers[idx])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    # plot the decision surface
    x_min, x_max = x.min()-1, x.max()+1
    y_min, y_max = y.min()-1, y.max()+1
    z_min, z_max = z.min()-1, z.max()+1

    resolution = 2.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    lr = [c for c in classifier.estimators_ if type(c) == LogisticRegression][0]
    clf_1 = lr
    zz = lambda x, y: (-clf_1.intercept_[0] - clf_1.coef_[0][0] * x - clf_1.coef_[0][1]) / clf_1.coef_[0][2]

    #val = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).T
    #Z = clf.predict(val)
    #Z = Z.reshape(xx.shape)

    # Plot stuff.
    ax.plot_surface(xx, yy, zz(xx, yy))

    plt.show()
