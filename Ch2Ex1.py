#from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

import numpy as np
y = df.iloc[0:100, 4].values # Create array of first 100 rows with only column 4, i.e. the correct label
y = np.where(y == 'Iris-setosa', -1, 1) # If
X = df.iloc[0:100, [0,2]].values # Get first 100 rows and make a new array with only position 0 and 2 as columns
plt.figure(1)
plt.scatter(X[:50, 0], X[:50,1], color='red', marker='o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

# plot errors over epochs
import Perceptron as p
plt.figure(2)
ppn = p.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()




import BoundaryMap
plt.figure(4)
BoundaryMap.plot_decision_regions(X, y, classifier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
