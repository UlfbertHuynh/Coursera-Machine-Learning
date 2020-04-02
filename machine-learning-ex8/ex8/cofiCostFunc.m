function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
rlist = [];
for i=1:num_movies
	for j=1:num_users
		if(R(i, j) == 1)
			rlist = [rlist; i j Y(i, j)];
		endif
	end
end

rlist = [rlist sum(Theta(rlist(:, 2), :) .* X(rlist(:, 1), :), 2) - rlist(:, 3)];

J_noreg = sum(rlist(:,4).^2);
reg_user = lambda * sum(sum(Theta.^2));
reg_movie = lambda * sum(sum(X.^2));
J = (J_noreg + reg_user + reg_movie) /2;

for i=1:num_movies
	for j=1:size(rlist, 1)
		if i == rlist(j,1)
			scalar = rlist(j, 4);
			vec = Theta(rlist(j, 2),:);
			X_grad(i,:) = X_grad(i,:) + vec*scalar;
		endif
	end
	X_grad(i,:) = X_grad(i,:) + lambda*X(i,:);
end

for i=1:num_users
	for j=1:size(rlist, 1)
		if i == rlist(j,2)
			scalar = rlist(j, 4);
			vec = X(rlist(j, 1),:);
			Theta_grad(i,:) = Theta_grad(i,:) + vec*scalar;
		endif
	end
	Theta_grad(i,:) = Theta_grad(i,:) + lambda*Theta(i,:);
end




%for i=1:num_users
%	map = rlist(rlist(:,2)==i,:);
%	scalars = map(:, 4);
%	vecs = X(map(:,1),:);
%	Theta_grad(i,:) = sum(vecs.*scalars);
%end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
