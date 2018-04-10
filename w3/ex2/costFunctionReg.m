function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
l = lambda;
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

X_theta_transp=X*theta; % X:m*n theta:n*1 -> X_theta_transp: m*1
h_theta=sigmoid(X_theta_transp); % h_theta: m*1
J_unregularized=sum((-y).*log(h_theta)-(1-y).*log(1-h_theta))/m; % J_unregularized: scalar
theta0=theta;
regularization=lambda*sum(theta(2:n).^2)/(2*m); % reg: scalar
J=J_unregularized+regularization; % J: scalar

%grad=(sum(X.*(h_theta .- y)))/m;
%for j=2:n
%  grad(j) = grad(j)+theta(j)*lambda/m;
%endfor

%reg_derivative=lambda*[0; theta(2:n)]/m;
%grad_without_reg=X'*(h_theta - y)/m;
%grad=grad_without_reg + reg_derivative;
grad=( X'*(h_theta - y) + lambda*[0; theta(2:n)] ) / m;


% =============================================================

end
