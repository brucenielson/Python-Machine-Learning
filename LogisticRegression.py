function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i = 1:size(z, 1);
  for j = 1:size(z, 2);
    g(i, j) = 1 / (1 + (e ^ -z(i,j)));
  end
end


% =============================================================

end




function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters.
%               You should set p to a vector of 0's and 1's
%

p = sigmoid(X * theta)>=0.5;





% =========================================================================


end




function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

# Get Cost
# Create vector with h(x) for each row in X
hypo = sigmoid(X * theta);
# Sum results using cost function for logistic regression
cost = sum( -y .* log(hypo) - (1 - y) .* log(1 - hypo) ) / m;
# Regularized sum should not include theta0 which is theta(1)
reg = (lambda / (2*m)) * sum(theta(2:length(theta)) .^ 2);
# Add in regularization sum to get a result that doesn't overfit
J = cost + reg;

# Now do the same for gradient, which is a vector the same size as theta (i.e. number of paramaters)
grad = (X' * (hypo - y)) ./ m;
# Create regularized sum, but skip theta(1)
for i = 2:length(theta)
  grad(i) = grad(i) + (lambda * theta(i) / m);
end
# Can this be vectorized?  YES! See lrCostFunction on ex3


% =============================================================

end




function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

# −y(i) log(hθ(x(i)))−(1−y(i))log(1−hθ(x(i)))



# Get Cost
# Create vector with h(x) for each row in X
hypo = sigmoid(X * theta);
# Sum results using cost function for logistic regression
J = sum( -y .* log(hypo) - (1 - y) .* log(1 - hypo) ) / m;


# Gradient for cost function
#for i = 1:size(X, 2); # Run through each theta in a vector of thetas
#  grad(i) = sum((hypo - y) .* X(:, i)) / m;
#end

# Vectorized version
grad = (X' * (hypo - y)) ./ m;





% =============================================================

end
