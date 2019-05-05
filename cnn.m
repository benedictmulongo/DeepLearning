
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
% F = ones(3,3);
% F(:,2) = 0*F(:,2);
% F(:,3) = -1*F(:,3);
% F

X = [[1 2 3 1];[4 5 6 1];[7 8 9 1]];
F = ones(2,2);
F1 = ones(2,2);
F1(1,1) = 0;
% vecF = [F(:);F1(:)];
vecF = ones(2,2,2);
vecF(:,:,1) = F;
vecF(:,:,2) = F1;

% X = randi([1 20],4,4);
% F = ones(3,2);

% M = matrixShit(X, F)
% [Stump,Shrink] = calculateStump(X, F)
% X = [[1 3 5 2 6 1];[7 0 2 4 1 3];[5 1 3 6 2 7];[2 4 3 0 0 1];[1 7 3 9 2 1];[3 6 2 4 1 0]]
% F = ones(3,3);
% F(:,2) = 0*F(:,2);
% F(:,3) = -1*F(:,3);
% F
M = matrixShitV3(X, F);
[RowC,ColC] = ConvDimension(X, F, 0, 1);
Result = M * X(:)
reshape(Result,[ColC,RowC])';

M = matrixShitV3(X, F1);
[RowC,ColC] = ConvDimension(X, F1, 0, 1);
Result2 = M * X(:);
reshape(Result2,[ColC,RowC])';

nFilters = 2;

M2 = MxMultipleFiltersV1(X, vecF, nFilters);
M4 = MxMultipleFiltersV2(X, vecF, nFilters);
Out = M4*X(:);
[lo,~] = size(Out);
c = lo/nFilters;
reshape(Out, [nFilters,c])

end 



function [] = testMFmatrixMultipleFilter()


X = [[1 2 3 1];[4 5 6 1];[7 8 9 1]];
F = ones(2,2);
F1 = ones(2,2);
F1(1,1) = 0;
vecF = ones(2,2,2);
vecF(:,:,1) = F;
vecF(:,:,2) = F1;

M = matrixShitV3(X, F);
[RowC,ColC] = ConvDimension(X, F, 0, 1);
Result = M * X(:);
reshape(Result,[ColC,RowC])';
nFilters = 2;

M2 = MxMultipleFiltersV1(X, vecF, nFilters)

% Mx1 = matrixShitV3(X, F);
% Mx2 = matrixShitV3(X, F1);
% [p1,p2] = size(Mx1);
% vecF1 = ones(p1,p2,2);
% vecF1(:,:,1) = Mx1;
% vecF1(:,:,2) = Mx2;
% 
% 
% M3 = MixTapeFilters(vecF1,2)


M4 = MxMultipleFiltersV2(X, vecF, nFilters)

end 

function [s] = vectorizeMultipleFilters(F,nFilters)

s = [];
for i = 1:nFilters
    b = F(:,:,i);
    s = [s;b(:)];
end

end 

function [] = testKroenerMultipleFilter()

X = [[1 2 3 1];[4 5 6 1];[7 8 9 1]];
F = ones(2,2);
F1 = ones(2,2);
F1(1,1) = 0;
vecF = [F(:);F1(:);F(:)];

nFilter = 3;
PP1 = convolutionMatrix(X,F,nFilter)
PP2 =convolutionMatrixKroenecer(X,F,nFilter)
size(PP2)
size(vecF)

Out = PP2*vecF
[lo,~] = size(Out);
c = lo/nFilter;
reshape(Out, [nFilter,c])

end 

function [RowC,ColC] = ConvDimension(X, F, pad, stride)

[RowX,ColX] = size(X);
[RowF,ColF] = size(F);

RowC = floor((RowX + 2*pad - RowF) / stride ) + 1;
ColC = floor((ColX + 2*pad - ColF) / stride ) + 1;

end 

function [S,S1] = calculateStump(X, F)

[n,~] = size(X);
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
[p,~] = size(F);
Ntimes_F_in_X = numberTimesFilterInX(X,F);
Ntimes_F_in_X_Row = numberTimesFilterInX(X(1:p,:),F);
M = zeros(Ntimes_F_in_X,n*m);
% [maxRow, maxCol] = size(M);
skip_N = n - p;

[~,Stump] = calculateStump(X, F);
length_stump = length(Stump);

Max = length(F(:)) + skip_N;

count = 0;
length_X = length(X(:));
cycle_detect = zeros(1,length_X);
Hasta = (Ntimes_F_in_X + Ntimes_F_in_X_Row)^2;
for index = 1:Hasta
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
   
    R = n * j ;
    R_mod_lengthX = mod(R,length_X);
    Update_index = R_mod_lengthX + cycle;
    if R_mod_lengthX + Max >= length_X
        continue
    end 

    M( count+1, (Update_index + 1):(Update_index + length_stump) ) = Stump;
    count = count + 1;
end

% size(M)
end 


function [M] = MxMultipleFiltersV1(X, F, nFilters)

[n,m] = size(X);
[p,~] = size(F(:,:,1));
Ntimes_F_in_X = numberTimesFilterInX(X,F(:,:,1));
Ntimes_F_in_X_Row = numberTimesFilterInX(X(1:p,:),F(:,:,1));
M = zeros(nFilters*Ntimes_F_in_X,n*m);
% [maxRow, maxCol] = size(M);
skip_N = n - p;

Stump = cell(1,nFilters);
for i=1:nFilters 
    [~,paste] = calculateStump(X, F(:,:,i));
    Stump{i} = paste;
end


length_stump = length(Stump{1});
Filter = F(:,:,1);
Max = length(Filter(:)) + skip_N;

count = 0;
length_X = length(X(:));
cycle_detect = zeros(1,length_X);
Hasta = (Ntimes_F_in_X + Ntimes_F_in_X_Row)^nFilters;
for index = 1:Hasta
    if count + 1> nFilters*Ntimes_F_in_X 
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
   
    R = n * j ;
    R_mod_lengthX = mod(R,length_X);
    Update_index = R_mod_lengthX + cycle;
    if R_mod_lengthX + Max >= length_X
        continue
    end 
    for k=1:nFilters 
        M( count + k, (Update_index + 1):(Update_index + length_stump) ) = Stump{k};
    end
    count = count + nFilters;
end

% size(M)
end 


%%%
function [O] = MxMultipleFiltersV2(X, F, nFilters)

[n,m] = size(X);
Ntimes_F_in_X = numberTimesFilterInX(X,F(:,:,1));

Mx_i = zeros(Ntimes_F_in_X,n*m, nFilters);


for i=1:nFilters 
    Mx_i(:,:,i) = matrixShitV3(X, F(:,:,i));
end

O = MixTapeFilters(Mx_i,nFilters);

end 
%%%

function [outPut] = MixTapeFilters(F,nFilters)


% [n,m] = size(X);
% Ntimes_F_in_X = numberTimesFilterInX(X,F(:,:,1));
% nFilters*Ntimes_F_in_X 
% M = zeros(nFilters*Ntimes_F_in_X,n*m);

[row,col] = size(F(:,:,1));
outPut = [];

for r = 1:row 
    temp = [];
    for i = 1:nFilters
        Fx = F(r,:,i);
        temp = [temp; Fx];
    end
    outPut = [outPut; temp];
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

function [OUT] = convolutionMatrix(X,F, n)

M = im2col(X, size(F))';
[nRow,nCol] = size(M);
I_bizarr = IdentityRaro(length(F(:)), n, n*nCol, 1);

% Output = zeros(n*nRow,n*nCol);
% Output = [];
% for i = 1:n*nRow
Output = cell(nRow,1);
for i = 1:nRow
    s = IdentityRaro(length(F(:)), n, n*nCol, M(i,:));
    Output{i} = s; 
%     Output(i,:) = s; 
%     Output = [Output; s]; 
    
end
OUT = cell2mat(Output);

end 




function [OUT] = convolutionMatrixKroenecer(X,F, n)

M = im2col(X, size(F))';
[nRow,nCol] = size(M);
Output = cell(nRow*n,1);
for i = 1:nRow
    K = kron(eye(n), M(i,:));
    Output{i} = K;     
end

% Output
OUT = cell2mat(Output);

end 

function [size_C] = numberTimesFilterInX(X,F)
% Estimate the number of times the Filter F can fit in X 
% used for the estimation of row for matrix Mx,nlen
B = im2col(X,size(F));
C = B'*F(:);
[size_C,~] = size(C);

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

