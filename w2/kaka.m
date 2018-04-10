function [y1,y2] = kaka(X,Z)

y1 = X*pinv(Z);
y2= X'*pinv(Z')
