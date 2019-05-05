
function [] = ExtractNames() 

data_fname = 'ascii_names.txt';

fid = fopen(data_fname,'r');
S = fscanf(fid,'%c');
fclose(fid);
names = strsplit(S, '\n');
if length(names{end}) < 1        
    names(end) = [];
end
name_length = zeros(length(names), 1);
labels = zeros(length(names), 1);
all_names = cell(1,length(names));
for i=1:length(names)
    nn = strsplit(names{i}, ' ');
    l = str2num(nn{end});
    if length(nn) > 2
        name = strjoin(nn(1:end-1));
    else
        name = nn{1};
    end
    name = lower(name);
    labels(i) = l;
    all_names{i} = regexprep(name,'\W','');
    name_length(i) = length(all_names{i}); 
end

disp('Saving the data')
tic
save('namesDataset.mat', 'labels', 'all_names', 'name_length');
toc

end