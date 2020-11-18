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

% Input layer
X = [ones(m, 1) , X] ;

% Hidden layer
Z2 = X * Theta1' ;
A2 = sigmoid(Z2) ;
A2 = [ones(size(A2 ,1), 1), A2] ;

% Output layer
Z3 = A2 * Theta2' ;
A3 = sigmoid(Z3) ;

counter = 1 : num_labels ;
y = (y == counter) ;

% without Regularization
for c = 1 : num_labels
   J = J + -( 1 / m) * sum( (y(:, c) .* log(A3(:, c))) + ( (1 - y(:, c)) .* log(1 - A3(:, c))) ) ;
endfor

% removing the bias terms from Theta
% adding regularization
Theta1_no_bias = Theta1(:, 2:end) ;
Theta2_no_bias = Theta2(:, 2:end) ;
J_reg = (lambda / (2 * m)) * ( sum(sum(Theta1_no_bias .* Theta1_no_bias, 1)) + sum(sum(Theta2_no_bias .* Theta2_no_bias, 1))) ;
J = J + J_reg ;

%
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


   for i = 1 : m

    % Input layer
    a1 = X(i, :)' ;  % 401 x 1 , column vector

    % Hidden layer
    z2 = Theta1  * a1 ; % 25 x 1
    a2 = sigmoid(z2) ;
    a2 = [1 ; a2] ; % 26 x1

    % Output layer
    z3 = Theta2  * a2 ; % 10 x 1
    a3 = sigmoid(z3) ;   % 10 x 1

    % calculating the error - layer 3
    delta_3 = a3 - y(i, :)'  ; % 10 x 1

    % adding a bias term to z2
    z2 = [1; z2] ; % 26 x 1

    % calculating the error - layer 2
    delta_2 =  (Theta2' * delta_3) .* sigmoidGradient(z2) ;
    delta_2 = delta_2(2:end); % remove bias term.  25 x 1

    %Accumulating
    Theta1_grad = Theta1_grad + (delta_2 *  a1') ;
    Theta2_grad = Theta2_grad + (delta_3 *  a2') ;

   endfor

% regularized gradient
Theta1_grad = ((1 / m) .* Theta1_grad) + ((lambda / m) .*  [zeros(size(Theta1_no_bias,1),1), Theta1_no_bias]) ;
Theta2_grad = ((1 / m) .* Theta2_grad) + ((lambda / m) .*  [zeros(size(Theta2_no_bias,1),1), Theta2_no_bias]) ;;

grad = [Theta1_grad(:) ; Theta2_grad(:)] ;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
