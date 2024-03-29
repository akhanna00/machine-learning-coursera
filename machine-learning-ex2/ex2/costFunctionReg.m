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

    h = sigmoid(X*theta); % hypothesis function
    matToBeSummed = -y.*log(h)-(1-y).*log(1-h); % the stuff in the inner sigma
    J = sum(matToBeSummed)/m; % compute the cost function
    reg = (lambda/(2*m)) * sum(theta(2:end).^2); % compute regularization factor
    J = J+reg; % add to cost
    
    % 
    diffInResult = h - y; % m by 1 vector
    grad = (X'*diffInResult)/m; % X is m by 26 matriz so we get a 26x1 vector corresponding to theta
    regGrad = zeros(size(theta));
    regGrad(2:end) = (lambda/m) * theta(2:end); % exclude theta1
    grad = grad + regGrad; % add to grad
    
    
    







% =============================================================

end
