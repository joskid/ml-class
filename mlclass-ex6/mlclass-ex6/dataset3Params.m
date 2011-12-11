function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sig_vals = c_vals;

best_C = 0;
best_sigma = 0;
old_error = 10000;

errors = zeros(length(c_vals),length(c_vals));

for i=1:length(c_vals)
  for j = 1:length(sig_vals)
    C = c_vals(i);
    sigma = sig_vals(j);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model,Xval);
    error = mean(double(predictions ~= yval));
    if error < old_error
      best_C = C;
      best_sigma = sigma;
      old_error = error;
    errors(i,j) = error;
  end;
end;

# errors

C = best_C;
sigma = best_sigma;
#min(errors)


% =========================================================================

end
