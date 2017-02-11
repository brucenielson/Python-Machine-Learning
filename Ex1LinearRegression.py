import matplotlib.pyplot as plt
import pandas
import numpy as np
import LinearRegression as lr
# from importlib import reload


# from imp import reload
# Clear and Close Figures
plt.close()

print('Loading data ...\n')

# Load Data
data = pandas.read_csv('ex1data2.txt', header=None).as_matrix()
print(type(data))
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

# Print out some data points
print('First 10 examples from the dataset: \n')
#print(' x = [%.0f %.0f], y = %.0f \n' % X[0:10,:], y[0:10])
for i in range(0,10):
    print('x = [' + str(X[i][0]) + ' ' + str(X[i][1]) + '], y = ' + str(y[i]) )
input('Program paused. Press enter to continue.\n')

# Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = lr.featureNormalize(X)


# Add intercept term to X
new_col = np.ones((m, 1), float)
X = np.append(new_col, X, axis=1)


# ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent
theta = np.zeros((3, 1),float)
theta, J1 = lr.gradientDescentMulti(X, y, theta, alpha, num_iters)
#thetax, J2 = lr.gradientDescentMulti(X, y, [0, 0, 0], 0.03, num_iters)
#thetax, J3 = lr.gradientDescentMulti(X, y, [0, 0, 0], 0.1, num_iters)
#thetax, J4 = lr.gradientDescentMulti(X, y, [0, 0, 0], 0.3, num_iters)
#thetax, J5 = lr.gradientDescentMulti(X, y, [0, 0, 0], 1, num_iters)


# Plot the convergence graph
plt.figure(1)
x_axis = np.linspace(1,J1.size, J1.size).reshape(400)
y_axis = J1.reshape(400)
plt.plot(x_axis, y_axis) #, '-b', 'LineWidth', 2

#hold on
#plot(1:numel(J2), J2, '-r', 'LineWidth', 2);
#plot(1:numel(J3), J3, '-k', 'LineWidth', 2);
#plot(1:numel(J4), J4, '-m', 'LineWidth', 2);
#plot(1:numel(J5), J5, '-c', 'LineWidth', 2);

plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()


# Display gradient descent's result
print('Theta computed from gradient descent: \n')
for item in theta:
    print(item)



# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

price = np.matmul(np.array([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]]).reshape(1,3), theta.reshape(3,1)) # You should change this


# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n' % price)

input('Program paused. Press enter to continue.\n')


# ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#

# Load data
data = pandas.read_csv('ex1data2.txt', header=None).as_matrix()
print(type(data))
X = data[:, 0:2]
y = data[:, 2]
m = len(y)


# Add intercept term to X
X = np.column_stack((np.ones((m, 1)), X))

# Calculate the parameters from the normal equation
theta = lr.normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
for item in theta:
    print(item)


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = np.matmul(np.array([[1, 1650, 3]]), theta)



#============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n' % price)

