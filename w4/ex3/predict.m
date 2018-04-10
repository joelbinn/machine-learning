function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add bias to X
A1=[ones(size(X,1),1),X];
% Calculate unit values of hidden layer (a2)
A2=sigmoid(A1*Theta1');
% Add bias to a2 layer
A2_w_bias=[ones(size(A2,1),1),A2];
% Calculate unit values of output layer (a3)
A3=sigmoid(A2_w_bias*Theta2');
% Find the index of the elements with the highest value of the output layer
% The index corresponds to the label (i.e. the digit in the picture)
[v,p] = max(A3,[],2);








% =========================================================================


end
