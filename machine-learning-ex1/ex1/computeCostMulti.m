function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
    h = X * theta; % hyothesis function should be m by 1 matrix
    J = (1/(2*m))*sum((y-h).^2); % compute cost function using formula (1/2m)*Sigma(h(x_i)-y_i)^2)




% =========================================================================

end
