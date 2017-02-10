import numpy as np
# from importlib import reload


def computeCostMulti(X, y, theta):
    # COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    #   COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y) # number of training examples

    # Compute the cost of a particular choice of theta

    hypo = X * theta
    costs = (hypo - y) ^ 2
    return sum(costs)/(2*m)
    # Redo with a matrix version of this once I know it works



def featureNormalize(X):
    # FEATURENORMALIZE Normalizes the features in X
    # FEATURENORMALIZE(X) returns a normalized version of X where
    # the mean value of each feature is 0 and the standard deviation
    # is 1. This is often a good preprocessing step to do when
    # working with learning algorithms.
    # Returns X_norm, mu, sigma

    mu = np.zeros((1, np.size(X, axis=1)), float) # size(X, axis=1) returns number of columns
    sigma = np.zeros((1, np.size(X, axis=1)), float)

    # First, for each feature dimension, compute the mean
    # of the feature and subtract it from the dataset,
    # storing the mean value in mu. Next, compute the
    # standard deviation of each feature and divide
    # each feature by it's standard deviation, storing
    # the standard deviation in sigma.

    # Note that X is a matrix where each column is a
    # feature and each row is an example. You need
    # to perform the normalization separately for
    # each feature.

    for i in range(0, np.size(X, axis=1)): # For i = 1 to number of columns
      mu[0][i] = np.mean(X, axis=0)[i] # find mean of each column and create a vector of those means
      sigma[0][i] = np.std(X, axis=0)[i] # find standard deviation for each column and store in a vector
      X_norm = (X - mu) / sigma # Return a new version of X that is element wise mean adjusted and standard deviation adjusted

    return X_norm, mu, sigma



def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha
    # Returns theta, J_history
    # Test Case:
    # Shape X is (47, 3)
    # Shape y is (47, 1)
    # Shape theta is (3,1)

    # Initialize some useful values
    m = np.size(y) # number of training examples
    J_history = np.zeros((num_iters, 1))
    summed = np.zeros((1, np.size(X, axis=1)))

    # Run through # of iterations
    for iter in range(0, num_iters):
        # Perform a single gradient step on the parameter vector theta.
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.

        hypo = 	np.matmul(X, theta) # Shape (47, 1)
        for i in range(0, np.size(X, axis=1)): # iterate over each column
          value = (hypo - y) * X[:, i]
          summed[i] = sum(value) # sum columns
          theta[i] = theta[i] - alpha * (1/m) * summed[i]


        # Save the cost J in every iteration
        J_history[iter] = computeCostMulti(X, y, theta)
        return theta, J_history



def normalEqn(X, y):
    # NORMALEQN Computes the closed-form solution to linear regression
    #   NORMALEQN(X,y) computes the closed-form solution to linear
    #   regression using the normal equations.
    # Returns theta
    theta = np.zeros((np.size(X, axis=1), 1))

    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.

    #θ =(XT*X)−1*XT~y.
    X_trans = np.transpose(X)
    theta = np.linalg.pinv(np.matrixmultipy(np.matrixmultipy(X_trans, X),  np.dot(X_trans * y)))
    return theta

