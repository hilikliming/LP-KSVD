%% This is a script for testing and comparing the performance of the LP-KSVD
% method versus the K-SVD and standard SVD method for manifold
% approximation given AC data signal type
clear all;
clc;
home =cd;
%% Defining Structs
Ytrain = struct([]);
Ytest = struct([]);
%% Opening the 5 designated FRM templates 
trainList = {'alcyl_2ft_1530','alcyl_3ft','alpipe','aluxo','bullet_105mm_air',...
    'bullet_105mm_h2o', 'howitzer_cap_air','howitzer_cap_h2o','howitzer_nocap','ssuxo'};
i = 1;
% Extracting and preprocessing training templates
for tag = trainList
cd(['C:\Users\halljj2\Desktop\WMSC-CODE\DBPond3.8','\ENV_1\',char(tag),'\10_m\90_deg']);
x = what;
x = x.mat;
ob = open(char(x));
AC = ob.acp';
cd(home);
Ytrain(i).D = formatAC(AC,301); % This script outputs target strength
i = i +1;
end
cd(home);


%% Training signal subspaces via SVD/K-SVD/LP-KSVD with same run parameters
% Adding clutter class to the Training Dictionaries

param.K                     = 40;
param.numIteration          = 60; % number of iterations to perform (paper uses 80 for 1500 20-D vectors)
param.preserveDCAtom        = 0;
param.InitializationMethod  = 'DataElements';
param.displayProgress       = 1;
param.minFracObs            = -.1; % min % of observations an atom must contribute to, else it is replaced
param.maxIP                 = 1-1e-8; % maximum inner product allowed betweem atoms, else it is replaced
param.coeffCutoff           = 1; % cutoff for coefficient magnitude to consider it as contributing

% Parameters related to sparse coding stage
coding.method = 'MP';
coding.errorFlag = 1;            
coding.errorGoal = 1e-2; % 1e-4 % allowed representation error for each signal (only if errorFlag = 1)
coding.denoise_gamma = 0.1;
coding.L = param.K;
coding.tau = 40;
coding.eta = 1e-4;

D_SVD = struct([]);
D_KSVD = struct([]);
D_LP = struct([]);

for m = 1:length(trainList)
     D = Ytrain(m).D; 
     %D = D{1,1};
     [U,S,V] = svd(D, 'econ');
     D_SVD(m).D = U(:,1:param.K);
     %D_KSVD(m).D = KSVD(D,param,coding);
     D_LP(m).D = LPKSVD(D,param,coding);
end
save('D_SVD.mat','D_SVD');
%save('D_KSVD.mat','D_KSVD');
save('D_LP.mat','D_LP');


%% Opening 10 templates at 4.8 meters and adding a noise matrix
testList = {'alcyl_2ft_1530','alcyl_3ft','alpipe','aluxo','bullet_105mm_air',...
    'bullet_105mm_h2o', 'howitzer_cap_air','howitzer_cap_h2o','howitzer_nocap','ssuxo'};
i = 1;
Y = [];
t_Y =[];
% Extracting and preprocessing training templates
for tag = testList
cd(['C:\Users\halljj2\Desktop\WMSC-CODE\DBPond4.8','\ENV_1\',char(tag),'\15_m\90_deg']);
x = what;
x = x.mat;
ob = open(char(x));
AC = ob.acp';
cd(home);
Ytest(i).Obs = formatAC(AC,301); % This script outputs target strength
Ytest(i).Obs = Ytest(i).Obs + 1/10*randn(size(Ytest(i).Obs)); % This script outputs target strength
Y = [Y, Ytest(i).Obs];
t_Y = [t_Y',i*ones(1,size(Ytest(i).Obs,2))]';
i = i +1;
end

cd(home);

%% Running the WMSC with Various Dictionaries
est = 'MSD';
sigA= 1;
d_YSVD = WMSC(Y,D_SVD,mu_m,est,sigA);
d_YKSVD = WMSC(Y,D_KSVD,mu_m,est,sigA);
d_YLP = WMSC(Y,D_LP,mu_m,est,sigA);

%% Displaying Results comparison


