import numpy as np
# from importlib import reload


def computeCostMulti(X, y, theta):
    # COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    #   COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y) # number of training examples
    f = np.size(X, axis=1)  # of features
    theta = theta.reshape((f, 1))  # Force theta into shape(f,1)
    y = y.reshape(m, 1) # Make sure y is a matrix shape instead of an array

    # Compute the cost of a particular choice of theta
    hypo = np.matmul(X, theta) # shape(m,f) * (f,1) = (m,1)
    costs = np.power((hypo - y), 2) # shape(m,1)
    return sum(costs)/(2*m) # final cost is a single number
    # Redo with a matrix version of this once I know it works



def featureNormalize(X):
    # FEATURENORMALIZE Normalizes the features in X
    # FEATURENORMALIZE(X) returns a normalized version of X where
    # the mean value of each feature is 0 and the standard deviation
    # is 1. This is often a good preprocessing step to do when
    # working with learning algorithms.
    # Returns X_norm, mu, sigma

    f = np.size(X, axis=1)
    X_norm = np.zeros(np.size(X))
    mu = np.zeros((1, f), float) # size(X, axis=1) returns number of columns: shape(1,features)
    sigma = np.zeros((1, f), float) # shape(1, features)

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

    for i in range(0, f): # For i = 1 to number of columns
      mu[0,i] = np.mean(X, axis=0)[i] # find mean of each column and create a vector of those means
      sigma[0,i] = np.std(X, axis=0)[i] # find standard deviation for each column and store in a vector

    X_norm = (X - mu) / sigma # Return a new version of X that is element wise mean adjusted and standard deviation adjusted

    return X_norm, mu.reshape(f,1), sigma.reshape(f,1)



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
    f = np.size(X, axis=1) # of features
    y = np.array(y).reshape((m,1)) # Change y shape from (47,) to (47,1) so I can work with it as a matrix easier.
    J_history = np.zeros((num_iters, 1), float) # Shape(400,1)
    summed = np.zeros((f,1), float) # Shape(f,1) which is the sum of each column (i.e. feature for each example)
    theta = np.reshape(theta, (f,1)) # Force theta into shape(f,1)

    # Run through # of iterations
    for iter in range(0, num_iters):
        # Perform a single gradient step on the parameter vector theta.
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.

        hypo = 	np.matmul(X, theta) # shape(samples, features) * shape(features, 1) = shape(samples, 1)
        diff = hypo - y # difference between the hypothesis and the real answers. shape(47,1)

        #for i in range(0, np.size(X, axis=1)): # iterate over each column
        #  Xcol = X[:,i].reshape(m,1)
        #  value = diff * Xcol
        #  summed[i] = sum(value) # sum columns
        #  theta[i] = theta[i] - alpha * (1/m) * summed[i]

        # Matrix version
        summed = np.sum(X * diff, axis=0).reshape(f, 1) # Shape(features, 1) i.e. (3, 1)
        theta = theta - alpha * (1/m) * summed

        # Save the cost J in every iteration
        J_history[iter] = computeCostMulti(X, y, theta)

    return theta, J_history



def normalEqn(X, y):
    # NORMALEQN Computes the closed-form solution to linear regression
    #   NORMALEQN(X,y) computes the closed-form solution to linear
    #   regression using the normal equations.
    # Returns theta

    # make sure inputs are numpy arrays
    X = np.array(X)
    m = np.size(X, axis=0) # Get number of training examples
    f = np.size(X, axis=1) # Get number of features
    y = np.array(y).reshape((m,1))

    theta = np.zeros((np.size(X, axis=1), 1))

    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.

    #θ =(XT*X)−1*XT~y.
    # theta = pinv(X'*X)*X' * y
    X_trans = X.T
    part1 = np.linalg.pinv(np.matmul(X_trans,X))
    part2 = np.matmul(part1, X_trans)
    theta = np.matmul(part2, y)
    #theta = np.linalg.pinv(np.matmul(np.matmul(X_trans, X),  np.matmul(X_trans, y)))
    return theta.reshape(f,1)

