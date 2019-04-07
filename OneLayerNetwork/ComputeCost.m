function J = ComputeCost(X,Y,W,b,lambda)

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


function P = evaluateClassifier(x,W,b)

s = W*x + b;
P = softmax(s);

end 