

function [X,Y,y,W,b,P,J] = softmaxClassifier()
rng(400)
% lambda = 0.001;
lambda = 0;
A = load('data_batch_1.mat');
Val = load('data_batch_2.mat');
test = load('test_batch.mat');

[X_val,Y_val,y_val,N_val] = loadBatch(Val);
[X_test,Y_test,y_test,N_test] = loadBatch(test);

[X,Y,y,N] = loadBatch(A);
[W,b] =  initialize(10,3072);
P = evaluateClassifier(X,W,b);
J = computeCost(X,Y,W,b, lambda);
GDparam = GDparams(100, 0.1, 40, N);
[Wstar, bstar] = miniBathGD(X, Y,X_val, Y_val,X_test,Y_test, GDparam, W, b, lambda);

size(Wstar)

for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

%  Print the learned weights 
figure 
montage(s_im, 'Size', [1,10])

% Calculate the difference between numerical and analytical gradient 

% Acc = computeAccuracy(X,Y,W,b);
% [grad_W,grad_b] = computeGradients(X,Y,P,W,lambda);

% s = 0;
% for t = 1:100
%     k =  randi(10000);
%     [grad_W,grad_b] = computeGradients(X(:,k), Y(:,k),P(:,k),W,lambda);
%     [grad_b1, grad_W1] = ComputeGradsNum(X(:,k), Y(:,k), W, b, lambda, 1e-6);
% %     size(X(:,1));
%     s = s +  gradientCheck(grad_W,grad_W1);
% 
% end 
% 
% the_sum_matrix = s 
end 

function [sum_matrix] =  gradientCheck(grad_W,grad_W1)

[row, col] = size(grad_W);
gradient_check = zeros(row,col);
for i = 1:row 
    for j = 1:col
        g_a = grad_W(i,j);
        g_n = grad_W1(i,j);
        gradient_check(i,j) = abs(g_a - g_n) / max(eps(1), abs(g_a) + abs(g_n));
    end
end

gradient_check = (gradient_check < 1e-6);
sum_matrix = sum(gradient_check,'all');

end 

function [X,Y,y,N] = loadBatch(file)

% Load the data set into in X -> image data
% Y -> one-hot labels 
% y -> true labels

K = 10;
[N,dim] = size(file.data);
y = file.labels + 1;
X = file.data';
Y = zeros(K,N);

X = double(X);
X = double(X)/255;

for i = 1:N
    index = y(i);
    Y(index,i) = 1;
end

end 

function [W,b] =  initialize(K,d)
W = normrnd(0,0.01,[K,d]);
b = normrnd(0,0.01,[K,1]);
end 


function p = softmaxh(n)

f = n - max(n);
p = exp(f)/ sum(exp(f));
end 

function P = evaluateClassifier(x,W,b)

s = W*x + b;
P = softmax(s);

end 

function J = computeCost(X,Y,W,b,lambda)

[dim,N] = size(X);
reg = 0;
for k = 1:10 
    for d = 1:dim
        reg = reg + (W(k,d)^2);
    end 
end 

L = 0;
for q = 1:N
    pi = evaluateClassifier(X(:,q),W,b);
    yi = Y(:,q);
    ri = -log(yi'*pi);
    L = L + ri;
    
end 

J = (1/N)*(L) + (lambda)*(reg);

end 

function accuracy = computeAccuracy(X,Y,W,b)

p = evaluateClassifier(X,W,b);
[argvalue, argmax] = max(p);

[S, N] = size(argmax);

accuracy = 0;
for i = 1:N
    index = argmax(i);
    result = Y(index, i);
    if result == 1
        accuracy = accuracy + 1 ;
    end 
end

accuracy = accuracy / N ;

end 


function [grad_W,grad_b] = computeGradients(X,Y,P,W,lambda)

[dim, N] = size(X);
G = -(Y - P);

lw_2 = 2 * lambda * W ;
grad_W = (1/N)* (G * X');
grad_b = (1/N)* (G * ones(N,1));
grad_W = grad_W + lw_2;

end 


function gdparam = GDparams(n_batch, eta, n_epochs,N)

gdparam.nbatch = n_batch ;
gdparam.eta = eta;
gdparam.n_epochs =  n_epochs ;
gdparam.N = N ;

end 


function [Wstar, bstar] = miniBathGD(X, Y,X_val, Y_val,X_test,Y_test, GDparams, W, b, lambda)

N = GDparams.N ;
n_batch = GDparams.nbatch;
eta = GDparams.eta;
n_epochs = GDparams.n_epochs ;

x_axis = 0:n_epochs -1;
y_axis = zeros(1,n_epochs);
yval_axis = zeros(1,n_epochs);

for iter= 1:n_epochs 
    for j=1:N/n_batch

        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        
        Xval_batch = X_val(:, j_start:j_end);
        Yval_batch = Y_val(:, j_start:j_end);

        P = evaluateClassifier(Xbatch,W,b);
        [grad_W,grad_b] = computeGradients(Xbatch, Ybatch,P,W,lambda);

        Jw = grad_W + 2*lambda*W ;
        Jb = grad_b ;

        W = W - eta * Jw ;
        b = b - eta * Jb ;

    end
    
    J = computeCost(Xbatch,Ybatch,W,b, lambda);
    y_axis(iter) = J;
    J2 = computeCost(Xval_batch,Yval_batch,W,b, lambda);
    yval_axis(iter) = J2;
    Acc = computeAccuracy(X,Y,W,b);
end
train_accuracy = computeAccuracy(X,Y,W,b)
test_accuracy = computeAccuracy(X_test,Y_test,W,b)
Wstar = W ;
bstar = b ;

% Create plot  
figure
plot(x_axis, y_axis)
hold on
plot(x_axis, yval_axis)

legend('Training loss','Validation loss' ,'Location', 'NorthWest')
xlabel('Epoch')
ylabel('Loss function')

end 

