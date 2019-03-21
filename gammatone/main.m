% Written by Zhang Wenyu.
%% Compute PF gammatone
tic;
clear;
list=dir(['../RWC_dataset_instrument/disk1/','01*.wav']);
for i = 1:length(list)
     % Extract the filename
     str = strcat ('../RWC_dataset_instrument/disk1/', list(i).name);
     [d,sr] = audioread(str);
     [D,~] = gammatonegram(d,sr);
     % Voice Activity Dectection (VAD)
     indhead = find(diff(sum(D==0))<-40);
     % Separate the .WAV file through VAD and store it in a cell matrix
     % 'check', where each element represents the gammatone matrix of a
     % single tone.
     check = mat2cell(D,size(D,1),diff([0,indhead,size(D,2)]));
     % Unite the dimension of 'check' in order to use the function @cellfun
     if (i>1)
         while(size(check,2)<size(data,2))
             check = [check,0];
         end
         while(size(check,2)>size(data,2))
             check = check(:,1:end-1);
         end
     end
     data(i,:) = check;
end
% Deal with 'data': if the column of an element of 'data' is less than some
% value 'k'(200, for example), then fill the missing blanks with 0;
% otherwise, extract the first k columns in that element.
data = cellfun(@extract,data,'UniformOutput',false);
% Delete the element that has only one row in 'data'
data = cellfun(@dele,data,'UniformOutput',false);
% Output 'data' in a 3D matrix.
data=reshape(data,1,[]);
data=data(1,cellfun(@(x) size(x,1)>1,data,'UniformOutput',true));
data=cell2mat(data);
for j = 1:(size(data,1)*size(data,2)/64/200)
    PF(j,:,:)=data(:,(200*j-199):(200*j));
end
toc;

save ../mat_files/PF.mat PF
