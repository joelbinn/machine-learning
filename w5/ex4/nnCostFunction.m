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

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25 x (400+1)
Theta2_grad = zeros(size(Theta2)); % 10 x (25+1)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
K_conv=eye(num_labels);
a_1=X;
z_2=[ones(size(a_1,1),1) a_1]*Theta1';
a_2=sigmoid(z_2);
z_3=[ones(size(a_2,1),1) a_2]*Theta2';
a_3=sigmoid(z_3);
h_theta=a_3;
J_unreg=-sum(sum(K_conv(y,:).*log(h_theta) + (1-K_conv(y,:)).*log(1-h_theta)))/m;
theta1_wo_1st_col_squared=[zeros(size(Theta1,1),1) Theta1(:,2:end)].^2;
theta2_wo_1st_col_squared=[zeros(size(Theta2,1),1) Theta2(:,2:end)].^2;
reg=lambda*(sum(sum(theta1_wo_1st_col_squared))+sum(sum(theta2_wo_1st_col_squared)))/(2*m);
J=J_unreg+reg;

K=(1:num_labels)';
DELTA_l_3=zeros(size(Theta2_grad)); % 10 x (25+1)
DELTA_l_2=zeros(size(Theta1_grad)); % 25 x (400+1)
for t = 1:m
  % 1. feed forward
  a_1_t=[1;a_1(t,:)']; % (400+1)x1
  a_2_t=[1;a_2(t,:)']; % (25+1)x1
  z_2_t=[1;z_2(t,:)'];
  a_3_t=a_3(t,:)'; % 10x1
  z_3_t=[1;z_3(t,:)'];
  % 2. calc delta
  y_t=y(t,:)';
  y_k_t=y_t == K;
  delta_k_3=a_3_t - y_k_t; % 10 x 1
  DELTA_l_3 = DELTA_l_3 + delta_k_3*a_2_t'; % 10 x 26 + 10 x 1 * 1 x 26 = 10 x 26
  delta_k_2=Theta2'*delta_k_3 .* sigmoidGradient(z_2_t); % 26 x 1
  DELTA_l_2=DELTA_l_2 + delta_k_2(2:end)*a_1_t'; % 25 x (400+1)  
endfor


% -------------------------------------------------------------

% =========================================================================
Theta1_grad=DELTA_l_2/m;
Theta1_wo_col1=[zeros(size(Theta1,1),1) Theta1(:,2:end)]/m;
Theta1_grad=Theta1_grad+lambda*Theta1_wo_col1;
Theta2_grad=DELTA_l_3/m;
Theta2_wo_col1=[zeros(size(Theta2,1),1) Theta2(:,2:end)]/m;
Theta2_grad=Theta2_grad+lambda*Theta2_wo_col1;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
