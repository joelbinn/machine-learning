function J = cost(X, y, theta)
m = size(X, 1);
pred = X*theta;
e_sqr=(pred-y).^2;
J = 1/(2*m)*sum(e_sqr);
