
function [] = createManipulateData()

[all_names, labels,alphabet,d,K,n_len] = initialize_params();
M = create_map(alphabet);

Matrix = load('namesMatrix.mat');
all_namesMatrix = Matrix.names_matrix;

size(all_namesMatrix)


end 


function [names_matrix] = all_names2matrix(all_names, M,d,n_len) 

names_matrix = zeros(d*n_len,numel(all_names) );

for j = 1:numel(all_names)
    
    first_name = all_names(j); 
    name_mat = name2matrix(first_name, M,d,n_len); 
    name_mat = name_mat(:);
    names_matrix(:,j) = name_mat;
    
end

save('namesMatrix.mat', 'names_matrix');

% Verify name 
% % % Matrix = load('namesMatrix.mat');
% % % all_namesMatrix = Matrix.names_matrix;
% % % nname = all_names(5)
% % % length(cell2mat(nname))
% % % [zeross,oness]=hist(all_namesMatrix(:,5),unique(all_namesMatrix(:,5)))
% % % 
% % % size(all_namesMatrix)


end 

function [name_mat] = name2matrix(the_name, M,d,n_len) 

nam = cell2mat(the_name);
name_mat = zeros(d,n_len);

for i = 1:numel(nam)
    character = nam(i);
    name_mat(:,i) = M(character)';
end

end 

function [all_names, labels,C,d,K,n_len  ] = initialize_params() 
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


function [M] = create_map(C) 

onehot_cell = cell(1,numel(C));
onehot = eye(numel(C),numel(C));
keyset = num2cell(C); 
% find(contains(keyset,'b'))

for i=1:numel(keyset)
    onehot_cell{i} = onehot(i,:);
end

M = containers.Map(keyset,onehot_cell);

end 