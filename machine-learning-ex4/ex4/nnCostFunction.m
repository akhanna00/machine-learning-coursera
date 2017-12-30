function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
       X = [ones(m, 1) X]; % add bias unit
       z_2 = X * Theta1'; 
       hiddenLayer = [ones(m, 1) sigmoid(z_2)]; % compute h(x) with bias unit
                                                        % for hidden layer
       z_3 = hiddenLayer * Theta2';                                         
       outputLayer = sigmoid(z_3); % compute output layer
       % compute the cost for each output unit for each example
       yOutputVectors = full(ind2vec(y', num_labels)); % expand out y so that it is
                                               % composed of 5000 vectors
                                               % with mostly 0s
       J = -sum(sum(yOutputVectors.*log(outputLayer') + (1-yOutputVectors).*log(1-(outputLayer'))))/m;
       
       % add reg factor to protect from overfitting
       reg = lambda*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))/(2*m);
       J = J + reg;
       
       
       




% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
    sum_delta_1 = zeros(size(Theta1));
    sum_delta_2 = zeros(size(Theta2));
    for i = 1:m
        yOutputVectorsT = yOutputVectors'; % compute transose
        y_i3 = yOutputVectorsT(i,1:end); % compute yVectors for 1 training example
        a_i3 = outputLayer(i, 1:end); % compute output vectors
        delta_3 = a_i3 - y_i3; % subtract to create 1x10 matrix
        delta_3 = delta_3'; % 10x1 vector
        error_2 = Theta2'*delta_3; % calculate the error
        error_2 = error_2(2:end,1); % remove the one corresponding to bias unit
        delta_2 = error_2.*sigmoidGradient(z_2(i,1:end))'; % multiply by sigmoid gradient
        % create running sum of deltas
        sum_delta_1 = sum_delta_1 + delta_2*X(i, 1:end);
        sum_delta_2 = sum_delta_2 + delta_3*hiddenLayer(i,1:end);
    end
    % divide by m examples
    Theta1_grad = sum_delta_1/m;
    Theta2_grad = sum_delta_2/m;
    

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
    % add reg parameter exluding first col
    Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda*Theta1(:,2:end)/m;
    Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda*Theta2(:,2:end)/m;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
