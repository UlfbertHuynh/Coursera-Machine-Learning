function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.2;
sigma = 0.1;

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
C_step = 0.2;
sigma_step = 0.1;

initial_C = C;
initial_sigma = sigma;
stop = 0;

model= svmTrain(X, y, initial_C + C_step, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
error = mean(double(predictions ~= yval));

while(stop == 0)
	model= svmTrain(X, y, initial_C + C_step, @(x1, x2) gaussianKernel(x1, x2, initial_sigma));
	predictions = svmPredict(model, Xval);
	error_new = mean(double(predictions ~= yval));
	if error_new > error
		stop = 1;
	else
		error = error_new;
		initial_C = initial_C + C_step;
	endif	
endwhile 

stop = 0;
C = initial_C;

while(stop == 0)
	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, initial_sigma + sigma_step));
	predictions = svmPredict(model, Xval);
	error_new = mean(double(predictions ~= yval));
	if error_new > error
		stop = 1;
	else
		error = error_new;
		initial_sigma = initial_sigma + sigma_step;
	endif	
endwhile
sigma = initial_sigma;


% =========================================================================

end
