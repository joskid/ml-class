function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%


initial_theta = zeros(n+1,1);
options = optimset('GradObj','on','MaxIter',50);

% For each class
for c = 1:num_labels
  % Get a vector of parameters which have the lowest cost to select
  % "1" on that class given the whole lot of inputs. So the first
  % pass will result in a parameter vector to classify images that
  % look like a "1" as 1 and everything else as 0.
  [theta] = fmincg(@(t)(lrCostFunction(t,X,(y==c),lambda)), ...
                   initial_theta,options);
  % Now turn that on its side and put it into the all_theta matrix
  % as the first row.
  all_theta(c,:) = theta';
  
end;

% Now all_theta is a matrix of K * n+1 size, representing K
% classifiers. Basically we want to use this with K
% classifiers. For the same inputs, on  K classifiers, the k-th
% classifier will output a 1 for inputs which match class k and 0
% for all other inputs.







% =========================================================================


end
