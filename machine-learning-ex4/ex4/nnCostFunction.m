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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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
X = [ones(m, 1) X];
a1 = X;
a2 = sigmoid (a1 * Theta1');
a2 = [ones(m , 1)  a2];
hx = sigmoid (a2 * Theta2');

% hx 是一个5000 * 10的矩阵，理解起来就是每一次输出是10个数字[1,0,0,0,...0]代表他认为是1，实际上hx出来的东西未必是整数（几乎可以肯定不是整数，算出来的只是最大可能性
% 因此hx实际上是[0.xx,0,0,0,0...0]）
% 那么在NN中，cost函数其实就是那10个输出的每一个的cost函数之和。
% 所以对于第一个（也就是认为其是1的可能的hx），我们首先取出hx矩阵里面，第一个输出的结果，因为总共有5000个训练内容所以是用hx(1:size(y), k)
% 那么y和hx其实是一模一样只是y是确定的输出值因为是训练的结果。
% 这里的y有点trickey，他的数据录入不是按照hx来的，而是按照最后所见所得的结果。
% 比如第一个输入案例是结果是5，那么y里面的第一个也是5，第二个是3，y里面也是3
% 所以y其实是类似于[1,3,4,5,5,7,3,1,7...]这样的。所以我们首先要和r去对比，还是以第一个输出节点来看
% 我们要关心的是他是1或者不是1，所以r = 1 * ones(size(y)),这样我们就得到一个[1,1,1,1,1,...]这样的向量。
% 进行一个==对比，就能得到所有是1的结果。
% 最后把所有的cost合起来就是整个nn的cost

costK = 0;
for (k = 1 : num_labels)
	r = k * ones(size(y));
	costK -= 1 / m * sum((r == y) .* log(hx(1:size(y), k)) + (1 - (r == y)) .* log(1 - ((hx(1:size(y), k)))));	
end
J = costK;

regTheta1 = Theta1(:, 2:end);
regTheta2 = Theta2(:, 2:end);

J = J + lambda / 2 / m * ((sum(sum(regTheta1 .^ 2))) + (sum(sum(regTheta2 .^ 2))));

ry = eye(num_labels)(y,:);
% 发现一个比较简单就能完成y的比对的方法, 这样就把y从5000 * 1，里面数据是直接显示1-10，变成了5000 * 10，在根据1-10决定某一列里面的值是1，其他都是0.

% 所以上面的方法也可以不用再用for循环
% 直接用矩阵的乘法
% cost矩阵：
% cost = ry .* log(hx) + (1 - ry) .* log (1 - hx);
% 然后求sum得时候有个要注意的，我们不能用sum(cost)，这样会把列的和加起来变成一个1 * 10的矩阵。
% 我们实际上需要的cost是和y的对比，我们通过ry把矩阵分解出来，所以需要把行加起来，所以用的是sum(cost,2)得到一个5000*10的矩阵。
% 最后再乘以-1/m
% J = -1 / m * sum(sum(cost, 2));
% 这和上面的结果是一样的。
delta3 =  hx - ry;
% δ(2) =  Θ(2) T δ(3). ∗ g′(z(2))
delta2 = (delta3 * Theta2)(:,2:end) .* sigmoidGradient(X * Theta1');

Delta1 = delta2' * a1;
Delta2 = delta3' * a2;
Theta1_grad = Delta1 / m + lambda / m * [zeros(hidden_layer_size , 1) Theta1(:,2:end)];
Theta2_grad = Delta2 / m + lambda / m * [zeros(num_labels , 1) Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
