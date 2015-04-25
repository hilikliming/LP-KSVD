function [ ACP, t_AC ] = formACP1(dir, names,aper)

home = cd;
% Directory which contains the SAS Images for each run...
cd(dir);
t_AC = [];
ACP =[];

for j = 1:length(names)
tag = char(names(j));
%tag = [tag '.mat'];
ob = load(tag);
samp = (ob.angles > -aper/2 & ob.angles < aper/2);
%Normalizing by to scale TS between -1 and 1 as was down by K-SVD 
img = ob.TS(samp,:);
img = 10.^(img/20);
for k = 1:size(img,1)
img(k,:) = img(k,:)/norm(img(k,:)); %Normalizing before reverting back to target strength
end
img = 20*log10(img);
t = j*ones(size(img,1),1);
t_AC = [t_AC; t];
ACP = [ACP img'];
end

cd(home);
end

