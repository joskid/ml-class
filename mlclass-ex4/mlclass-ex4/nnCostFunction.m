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

disp(size(Theta1));
disp(size(Theta2));

% Setup some useful variables
m = size(X, 1);
k = num_labels;
         
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

% First we need to turn our input labels into a classification
% matrix where each row represents the logical class of that
% feature, eg. [0 0 1 0 0 0 0 0 0 0] for a value of 3. How?

% This creates a m*k (5000*10) matrix which will be our new y.
% Then iterates over all labels 1-k and sets each column of maty to
% the result of a logical comparison "are any of the elements of y
% equal to this label?". So the 1st column of maty will have 1 for
% each training example of label 1, the 4th column of maty will
% be 1 for each example of label 4, etc.

maty = zeros(m,k);
for i = 1:k
  maty(:,i) = y == i;
end;

% We need the final hTHETA(X), so let's compute it.
% Set a1 to be X + bias unit
a1 = [ones(rows(X),1),X];

% This is Layer 2 of the network (the hidden layer)
z2 = a1*Theta1';
a2 = sigmoid(z2);
% a2 is a 5000 x 25 matrix

% Add the bias unit to a2
a2 = [ones(rows(a2),1),a2];

% Layer 3 of the network (the output layer)
z3 = a2*Theta2';
a3 = sigmoid(z3);

% a3 is now a 5000 x 10 matrix. (Just like y).
% a3(1,1) indicates the likelihood that feature 1 is in category 1.
% a3(1,7) is the probability that feature 1 is in category 7.
% To find the category of feature 1, we would find which column of
% the first row has the highest value - that would be our best
% guess. So if the highest value is in column 8, our model thinks
% that feature 1 is in class 8.
%
% So to sum over all K, we need to sum all the columns for each
% feature, then to sum over all m, we need to sum those rows.

% Calculate cost function of regularized logistic regression. This
% is vectorised implementation as outlined in the ex3.pdf notes. 

J = 1/m * sum(sum(-maty .* log(a3) - (1-maty) .* log(1-a3),2))

% Calculate the regularisation expression and add it to J
% obviously could be on one line but whatever
reg = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2,2))+ ...
                     sum(sum(Theta2(:,2:end).^2,2)));
J += reg;


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

% Calculate the gradients

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% OK let's ignore that and do what he says and use a for-loop.

for t = 1:m % So we'll be operating on row vectors
  a1 = [1,X(t,:)]'; % Note the transpose there
  z2 = Theta1 * a1;
  a2 = [1;sigmoid(z2)];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  d3 = a3 - maty(t,:)';
  d2 = (Theta2)'*d3 .* [1;sigmoidGradient([z2])]; % do NOT get why
                                                  % this works
  Delta2 += d3*a2';
  Delta1 += d2(2:end,:)*a1';
end;

Theta1_grad = 1/m*Delta1;
Theta2_grad = 1/m*Delta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad += [zeros(rows(Theta1_grad),1),lambda/m*Theta1(:,2: ...
                                                  end)];

Theta2_grad += [zeros(rows(Theta2_grad),1),lambda/m*Theta2(:,2: ...
                                                  end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
