function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Hypothesis of the params... a mini-function to make it make more
% sense in my head
h = @(theta,X) sigmoid(X*theta);

% Calculate cost function of regularized logistic regression. This
% is vectorised implementation as outlined in the ex3.pdf notes. 
J = 1/m * (sum(-y .* log(h(theta,X)) - (1-y) .* log(1-h(theta,X))))+ ...
    lambda/(2*m)*sum(theta(2:end).^2);
x = lambda / (2*m) * sum(sum(Theta1(:,2:end).^2,2)) + ...
    sum(sum(Theta2(:,2:end).^2,2));

J += x;
% Make a dodgy theta with 0 as 1st element (code for "ignore it").
% The 0 at the top will mean that when the lambda (regularisation)
% is added or subtracted, it will always be 0. Meaning that you
% will always be adding or subtracting 0 to x0, which is what we want.
thetar = [0;theta(2:end)];

% Use it to "cheat" to produce the gradients (derivatives?) of each
% row of X. This works but to be honest I'm not exactly sure why.

grad = 1/m * (X'*(h(theta,X)-y)+lambda*thetar);

% =============================================================

end
