
function [] = cnn()

[all_names, labels,alphabet,d,K,n_len] = initialize_variables() ;
Matrix = load('namesMatrix.mat');
X = Matrix.names_matrix;
n1 = 1;
k1 = 30;
n2 = 1;
k2 = 32;
ConvNet = initialize_hyperparams(n1, k1, n2, k2, d,K);


X1 = 20*ones(3,3);
X2 = zeros(3,3);
X3 = [X1,X2];
X = [X3;X3];
% X4 = [X2,X1];
% X = [X3;X4];

% F = ones(2,2);

% F = ones(3,3);
% F(2,:) = 0*F(2,:);
% F(3,:) = -1*F(3,:);
% X = [[1 3 5 2 6 1];[7 0 2 4 1 3];[5 1 3 6 2 7];[2 4 3 0 0 1];[1 7 3 9 2 1];[3 6 2 4 1 0]]
F = ones(3,3);
F(:,2) = 0*F(:,2);
F(:,3) = -1*F(:,3);
F

% X = [[1 2 3 1];[4 5 6 1];[7 8 9 1]];
% F = ones(2,2);+
% PP = MakeMFMatrix(F, X,1);
% X = randn(3,4);
% F = ones(2,2);

% M = matrixShit(X, F)
% [Stump,Shrink] = calculateStump(X, F)
M = matrixShitV3(X, F);
[RowC,ColC] = ConvDimension(X, F, 0, 1);
Result = M * X(:);
reshape(Result,[RowC,ColC])
reshape(Result,[ColC,RowC])'
% reshape(Result,[2,3])
% reshape(Result,[3,2])'
% matrixShitV2(X, F);
% convolutionMatrix(X,F,2)
% IdentityRaro(lengthF, row, col,1)
% IdentityRaro(8, 3, 24,1);

% P_size = size(PP)
% Result = PP * X(:)
% Result_size = size(Result)
% F = randn(4,2,2);
% MakeMFMatrixes(F, 4);

end 

function [RowC,ColC] = ConvDimension(X, F, pad, stride)

[RowX,ColX] = size(X);
[RowF,ColF] = size(F);

RowC = floor((RowX + 2*pad - RowF) / stride ) + 1;
ColC = floor((ColX + 2*pad - ColF) / stride ) + 1;

end 

function [S,S1] = calculateStump(X, F)

[n,m] = size(X);
[p,q] = size(F);
skip_N = n - p;

S = [];
skip = zeros(1,skip_N);
shrink = p*q + (q-1)*skip_N;
for i = 1:q
    K = F(:,i)';
    S = [S,K,skip];
end

S1 = S(:,1:shrink);

end 

function [M] = matrixShitV3(X, F)

[n,m] = size(X);
[p,q] = size(F);
Ntimes_F_in_X = numberTimesFilterInX(X,F);
Ntimes_F_in_X_Row = numberTimesFilterInX(X(1:p,:),F);
M = zeros(Ntimes_F_in_X,n*m);
[maxRow, maxCol] = size(M);
skip_N = n - p;

[~,Stump] = calculateStump(X, F);
length_stump = length(Stump);

Max = length(F(:)) + skip_N;

count = 0;
length_X = length(X(:));
cycle_detect = zeros(1,length_X);
Hasta = (Ntimes_F_in_X + Ntimes_F_in_X_Row)^2;
for index = 1:Hasta
% for index = 1:Ntimes_F_in_X
    if count + 1> Ntimes_F_in_X 
        break
    end
    j = index - 1;
    s = j*(p + skip_N) + 1;
    k = mod(s,length_X);
    cycle = cycle_detect(k);
    if cycle_detect(k) == 0
        cycle_detect(k) = 1;
    else 
        cycle_detect(k) = cycle_detect(k)+ 1;
    end
   
    the_index = j;
    R = n * j ;
    R_mod_lengthX = mod(R,length_X);
    Update_index = R_mod_lengthX + cycle;
    if R_mod_lengthX + Max + 1 >= length_X
        continue
    end 

    M( count+1, (Update_index + 1):(Update_index + length_stump) ) = Stump;
%     [~, testCol] = size(M);
%     if testCol > maxCol 
%     top_index_update = (Update_index + length_stump);
%     if top_index_update <= testCol 
%         M( count+1, (Update_index + 1):(Update_index + length_stump) ) = Stump;
%     end
    
%     size(M)
%     [the_index,R,R_mod_lengthX,Update_index]
    count = count + 1;
end

% Final_size = size(M)
% M = M(1:Ntimes_F_in_X,1:n*m);
% size(M)
end 

function [M] = matrixShitV2(X, F)

[n,m] = size(X);
[p,q] = size(F);
Ntimes_F_in_X = numberTimesFilterInX(X,F);
M = zeros(Ntimes_F_in_X,n*m);
skip_N = n - p;

[~,Stump] = calculateStump(X, F);
length_stump = length(Stump);

Max = length(F(:)) + skip_N;

count = 1;
length_X = length(X(:));
cycle_detect = zeros(1,length_X);
for index = 1:Ntimes_F_in_X
    j = index - 1;
    s = j*(p + skip_N) + 1;
    k = mod(s,length_X);
    limit = s + Max;
    before = [s,k,limit > length_X]
    
    cycle = cycle_detect(k);
    if cycle_detect(k) == 0
        cycle_detect(k) = 1;
    else 
        cycle_detect(k) = cycle_detect(k)+ 1;
    end
    
    
        

    if limit > length_X 
        pad = count + cycle;
        
%         s = (j + 1)*(p + skip_N) + count;
        s = (j + 1)*(p + skip_N) + pad;
        k = mod(s,length_X);
    end
    after = [s,k,limit > length_X ]
    
%     M[index, ]
    

end


M = 0;
end 


function [M] = matrixShit(X, F)

[n,m] = size(X);
[p,q] = size(F)
Ntimes_F_in_X = numberTimesFilterInX(X,F)
M = zeros(Ntimes_F_in_X,n*m);

skip = n - p


for i = 1:Ntimes_F_in_X
    ROW = i
    for j = 1:q
%         index1 = i;
%         index2 = ( j*p + j*skip );
%         index3 = ( (j+1)*p + j*skip );
%         [index1,index2,index3]
        index1 = i;
        index2 = ( (j-1)*p + (j-1)*skip ) + 1;
        index3 = ( j*p + (j-1)*skip ) ;
        [index1,index2,index3]
        M( i, index2:index3 ) = 1 ;
%         M( i, ( (j-1)*p + (j-1)*skip ) : ( j*p + (j-1)*skip ) ) = 1 ;
%         M( i, ( j*p + j*skip ) : ( (j+1)*p + j*skip ) ) = 1 ;
    end
end



end 

function [t] = IdentityRaro(lengthF, row, col, Value)

t = zeros(row,col);

offset = 0;
for i = 1:row
    A = (offset + 1);
    B = (lengthF + offset);
    t(i, A:B) = Value;
    offset = offset + lengthF;
end

end 

function [] = convolutionMatrix(X,F, n)

M = im2col(X, size(F))'
[nRow,nCol] = size(M);
I_bizarr = IdentityRaro(length(F(:)), n, n*nCol, 1)

% Output = zeros(n*nRow,n*nCol);
% Output = [];
% for i = 1:n*nRow
Output = cell(nRow,1);
for i = 1:nRow
    s = IdentityRaro(length(F(:)), n, n*nCol, M(i,:))
    Output{i} = s; 
%     Output(i,:) = s; 
%     Output = [Output; s]; 
    
end

cell2mat(Output)
OUT = Output

% Mb = eye(nCol * n)


end 

function [size_C] = numberTimesFilterInX(X,F)
% Estimate the number of times the Filter F can fit in X 
% used for the estimation of row for matrix Mx,nlen
B = im2col(X,size(F));
C = B'*F(:);
[size_C,~] = size(C);

end 

function [size_C] = trics(M,F)
% Estimate the number of times the vectorized
% version of Filter F can fit in M
% used for the estimation of number of step required,.
[k,q] = size(F);
B = im2col(M,[1,k*q]);
C = B'*F(:);
[size_C,~] = size(C);

end 



function [G] = MakeMFMatrix(F,X, nlen)

[dd, k] = size(F);
Row = (nlen - k + 1);
% Row = size_C;
Row = numberTimesFilterInX(X,F);
[a,b] = size(X);
Col = (a*b)

Mf_nlen = zeros(Row, Col);
ntimes = trics(Mf_nlen,F)/Row
vecF = F(:);
offset = 0;
% step = (Col -length(vecF)  )/ (ntimes - 1)
% nlen = (Col -length(vecF))/ (ntimes - 1)
for i = 1:Row
    
    if offset >= Col - nlen
        break;
    end
    
    A = (offset + 1);
    B = (length(vecF) + offset);
    Mf_nlen(i, A:B) = vecF;
    offset = offset + nlen;
end

G = Mf_nlen

end 

function [G] = MakeMFMatrixes(F, nlen)

[dd, k, N] = size(F);
Filters = cell(1,N);

S = [];
for i=1:N
    temp = F(:,:,i);
    S = [S; temp(:)' ];
    Filters{i} = F(:,:,i); 
end
S
Row = (nlen - k + 1)*N
Col = (nlen*dd)

Mf_nlen = zeros(Row, Col);
vecF = S;
offset = 0;
for i = 1:Row
%     index1 = [(N*(i-1) + 1),(N*i)]
%     index1 = [tempory,(tempory + N - 1)]
    if ((N*(i-1) + 1) + N - 1) <= Row  
        tempory = (N*(i-1) + 1);
        A = (offset + 1);
        B = (length(vecF) + offset);
        Mf_nlen(tempory:tempory + N - 1, A:B) = vecF;
        offset = offset + nlen;
    end 
end

% G = Mf_nlen(1:Row, 1:Col)
G = Mf_nlen
G_size = size(G)

end 

function [ConvNet] = initialize_hyperparams(n1, k1, n2, k2, d,K)

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% n1 -> The number of filters in layer 1
% k1 -> The width of the filter in layer 1
% n2 -> The number of filters in layer 2
% k2 -> The width of the filter in layer 2
% Filter 1 = n1 x (d X k1) and Filter 2 = n2 x (n1 X k2)
% HE initialization :
% w=np.random.randn(layer_size[l],layer_size[l-1])*np.sqrt(2/layer_size[l-1])
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sig1 = 1;
sig2 = sqrt(2/(n1));
sig3 = sqrt(2/(n2));

ConvNet.F{1} = randn(d,k1,n1)*sig1;
ConvNet.F{2} = randn(n1,k2,n2)*sig2;
figsize = n2; % No sure ?
ConvNet.W  = randn(K,figsize)*sig3;
ConvNet.eta = 0.001;
ConvNet.rho = 0.9;

end

function [all_names, labels,C,d,K,n_len  ] = initialize_variables() 
name = load('namesDataset.mat');
all_names = name.all_names;
labels = name.labels;
Q = name.name_length;
% Dictionary: all the unique characters used the name database
C = unique(cell2mat(all_names));
% Dimensionality 
d = numel(C);
% Number of class
K = numel(unique(labels));
% Length of the longest name 
[n_len,~] = max(Q);

end 

