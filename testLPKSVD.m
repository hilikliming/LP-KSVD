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
trainList = {'aluxo','ssuxo','alpipe','alcyl_2ft_1'};
i = 1;
% Extracting and preprocessing training templates
for tag = trainList
cd(['C:\Users\halljj2\Desktop\old WMSC_EX0910\DBFRM\ENV_1\',char(tag),'\10_m\90_deg']);
x = what;
x = x.mat;
ob = open(char(x));
AC = ob.acp';
cd(home);
Ytrain(i).D = formatAC(AC,301); % This script outputs target strength
i = i +1;
end
cd(home);


%% Opening 10 templates at 3.8 meters
i = 1;
Y = [];
t_Y =[];

cd(home);
% These structs hold the pond AC's
Tpond = [2,8,11]; % aluxo, realuxo, ssuxo
NTpond = [4,5,9,10]; % cylinder, pipe, rock1, rock2

DT_pond =struct([]);
DNT_pond = struct([]);
l = 1;

TemplateDir = 'C:\Users\halljj2\Desktop\WMSC-CODE\UW Pond\EXPERIMENT_TARGET_TEMPLATES';
cd(TemplateDir);
x = what;x=x.mat;
cd(home);
% Grabbing AC templates for various pond objects in T and NT classes
for t = Tpond
[DT_pond(l).D, ~] = formACP1(TemplateDir,x(t),720);
l = l+1;
end

% Real Non-target object dictionary formation
l = 1;
for t = NTpond
[DNT_pond(l).D, ~] = formACP1(TemplateDir,x(t),720);
l = l+1;
end
cd(home);

%% Partitioning Rock Data
% Shuffling Clutter Aspects
Dclutter = [DNT_pond(3).D,DNT_pond(4).D];
Dclutter = Dclutter(:,randperm(size(Dclutter,2)));

% Splitting Clutter samples
DcTrain = Dclutter(:,1:size(Dclutter,2)/2);
DcTest = Dclutter(:,size(Dclutter,2)/2+1:end);

%% Creating mu_m struct for our M different classes
mu_m = struct([]);

% 4 non-clutter object classes
% aluxo, ssuxo, alpipe, alcyl2ft1 
Dtot_FRM = {Ytrain(1).D,Ytrain(2).D,Ytrain(3).D,Ytrain(4).D};
% 4 non-clutter test objects
% aluxo, real uxo, ssuxo, pipe, cylinder
Dtot_pond = {DT_pond(1).D, DT_pond(3).D,DNT_pond(1).D, DNT_pond(2).D};
M = length(Dtot_pond);
% Calculating noise means for each of the T and NT object classes
for m = 1:M
    frmAC = Dtot_FRM(m);frmAC = frmAC{1,1};
    pondAC = Dtot_pond(m); pondAC = pondAC{1,1};
    mu_m(m).mu = mean(pondAC-frmAC,2);
end
mu_m(5).mu=mu_m(4).mu*0;

save('DcTrain.mat','DcTrain');
save('DcTest.mat','DcTest');
save('Dtot_FRM.mat','Dtot_FRM');
save('Dtot_pond.mat','Dtot_pond');
save('mu_m.mat','mu_m');

load('DcTrain.mat');
load('DcTest.mat');
load('Dtot_FRM.mat');
load('Dtot_pond.mat');
load('mu_m.mat');

% Grabbing the portions of Pond Data to construct observation signal
grab = [1:4]; % Taking all uxos and the cylinder plus the pipe observation
sub = Dtot_pond(grab);
Y0 = []; t_pond =[];
M = length(Dtot_FRM);

for n = 1:length(sub)
    ob = sub(n);
    Y0 = [Y0, ob{1,1}];
    t_pond = [t_pond; ones(size(ob{1,1},2),1)*n];
end

% Adding tested clutter to end of observation matrix
Y = [Y0 DcTest];
t_Y = [t_pond; ones(size(DcTest,2),1)*(length(sub)+1)];
origT = t_Y;

% UXO vs. non-UXO (last 3 are cylinder, pipe, rocks)
% First 3 are aluxo,realuxo, ssuxo
t_Y(t_Y==1|t_Y==2)= 1;
t_Y(t_Y~=1)= 0;


%% Training signal subspaces via SVD/K-SVD/LP-KSVD with same run parameters
% Adding clutter class to the Training Dictionaries


param.numIteration          = 100; % number of iterations to perform (paper uses 80 for 1500 20-D vectors)
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

coding.tau = 10;
coding.eta = 1e-2;

D_SVD = struct([]);
D_KSVD = struct([]);
D_LP = struct([]);
Ytrain(5).D=DcTrain;
DD = [];
ms= [100,100,100,100,100];
% Creating SVD and KSVD dictionaries and accumulating KSVD atoms for 
% LPKSVD joint solution

D_SVD=open('D_SVD.mat'); D_SVD=D_SVD.D_SVD;
D_KSVD=open('D_KSVD.mat'); D_KSVD=D_KSVD.D_KSVD;
Data = [];
for m = 1:length(Ytrain)
    D = Ytrain(m).D; 
    %param.K  = ms(m);
    %coding.L = param.K;
    %[U,S,V] = svd(D, 'econ');
    %D_SVD(m).D = U(:,1:ms(m));
    %D_KSVD(m).D = KSVD(D,param,coding);
    DD= [DD,D_KSVD(m).D];
    Data = [Data,D];
end

param.K= size(DD,2);
param.Dict = DD;
param.numIteration = param.numIteration*5;
DDD = LPKSVD(Data,param,coding);
b=1;
for m=1:length(Ytrain)
D_LP(m).D = DDD(:,b:b+ms(m)-1);
b= ms(m)+b;
end

save('D_SVD.mat','D_SVD');
save('D_KSVD.mat','D_KSVD');
save('D_LP.mat','D_LP');


D_LP=open('D_LP.mat'); D_LP=D_LP.D_LP;

%% Running the WMSC with Various Dictionaries
mSVD =  [61    65    71   100    46];
mKSVD = [51    55    71   100    56];
mLP =   [61    65    71   100    46];
for m =1:length(Ytrain)
    USVD = D_SVD(m).D;
    UKSVD = D_KSVD(m).D;
    ULP = D_LP(m).D;
    USVD = USVD(:,1:mSVD(m));
    UKSVD = UKSVD(:,1:mKSVD(m));
    ULP = ULP(:,1:mLP(m));
    D_SVD(m).D= USVD;
    D_KSVD(m).D= UKSVD;
    D_LP(m).D= ULP;
end
est = 'MSD';
sigA= 1;
d_YSVD = WMSC(Y,D_SVD,mu_m,est,sigA);
d_YKSVD = WMSC(Y,D_KSVD,mu_m,est,sigA);
d_YLP = WMSC(Y,D_LP,mu_m,est,sigA);

%% Displaying Results comparison

figure;
plot(d_YSVD);
title('SVD');
legend('J_1','J_2','J_3','J_4','J_C');
axis([0,size(Y,2),0,max(max(d_YSVD))]);
figure;
plot(d_YKSVD);
title('KSVD');
legend('J_1','J_2','J_3','J_4','J_C');
axis([0,size(Y,2),0,max(max(d_YKSVD))]);
figure;
plot(d_YLP);
title('LP-KSVD');
legend('J_1','J_2','J_3','J_4','J_C');
axis([0,size(Y,2),0,max(max(d_YLP))]);

%% Documenting Result/Performance for Analysis

% Initializing min discriminant value vector for K observations and a
% decision vector m_P
[N, K] = size(Y);

jmin_TSVD = zeros(K,1);
jmin_NTSVD = zeros(K,1);

jmin_TKSVD = zeros(K,1);
jmin_NTKSVD = zeros(K,1);

jmin_TLP = zeros(K,1);
jmin_NTLP = zeros(K,1);

m_YSVD = zeros(K,1);
m_YKSVD = zeros(K,1);
m_YLP = zeros(K,1);


% Determine Minimal Discriminant Value to make decision and record decision
T = [1,2];
NT = [3,4,5];

% Finding minimal value from UXO and non UXO families of classes
for k = 1:K
% Strict Class decision
[~, m_YLP(k)] = min(d_YLP(k,:));
jmin_TLP(k) = min(d_YLP(k,T));%/norm(d_Y(k,T));
jmin_NTLP(k) = min(d_YLP(k,NT));%/norm(d_Y(k,NT));
[~, m_YSVD(k)] = min(d_YSVD(k,:));
jmin_TSVD(k) = min(d_YSVD(k,T));%/norm(d_Y(k,T));
jmin_NTSVD(k) = min(d_YSVD(k,NT));%/norm(d_Y(k,NT));
[~, m_YKSVD(k)] = min(d_YKSVD(k,:));
jmin_TKSVD(k) = min(d_YKSVD(k,T));%/norm(d_Y(k,T));
jmin_NTKSVD(k) = min(d_YKSVD(k,NT));%/norm(d_Y(k,NT));
end

% Setting UXO class I for objects classified as 1,2 or 3, other non UXO
% detections become non UXO class 0
origMLP = m_YLP;
origMSVD = m_YSVD;
origMKSVD = m_YKSVD;

for t = T
m_YLP(m_YLP==t) = 1;
m_YSVD(m_YSVD==t) = 1;
m_YKSVD(m_YKSVD==t) = 1;
end
m_YLP(m_YLP~=1) = 0;
m_YSVD(m_YSVD~=1) = 0;
m_YKSVD(m_YKSVD~=1) = 0;

%% ROC Curve Forming
gammas  = 0:1e-3:3;
gammas  = [gammas, 100];

P_dLP     = zeros(length(gammas),1);
P_faLP    = P_dLP;

P_dSVD     = P_dLP;
P_faSVD    = P_dLP;

P_dKSVD     = P_dLP;
P_faKSVD    = P_dLP;

gam_i   = 1;

for gamma = gammas
    dLP = zeros(K,1);
    dSVD = zeros(K,1);
    dKSVD = zeros(K,1);
    for k = 1:K
     dLP(k) = (jmin_TLP(k)/jmin_NTLP(k) < gamma);
     dSVD(k) = (jmin_TSVD(k)/jmin_NTSVD(k) < gamma);%If the J_m that lies closest to m subspace is still too great we say it's a sounding
     dKSVD(k) = (jmin_TKSVD(k)/jmin_NTKSVD(k) < gamma);
    end
    dLP = logical(dLP);
    dSVD = logical(dSVD);
    dKSVD = logical(dKSVD);
    P_dLP(gam_i) = sum(dLP & logical(t_Y))/sum(logical(t_Y));
    P_faLP(gam_i) = sum(dLP & ~logical(t_Y))/sum(~logical(t_Y));
    P_dSVD(gam_i) = sum(dSVD & logical(t_Y))/sum(logical(t_Y));
    P_faSVD(gam_i) = sum(dSVD & ~logical(t_Y))/sum(~logical(t_Y));
    P_dKSVD(gam_i) = sum(dKSVD & logical(t_Y))/sum(logical(t_Y));
    P_faKSVD(gam_i) = sum(dKSVD & ~logical(t_Y))/sum(~logical(t_Y));
    gam_i = gam_i +1;
end

% Finding the knee point
[~,gamkLP] = min(abs(1-(P_dLP+P_faLP)));
[~,gamkSVD] = min(abs(1-(P_dSVD+P_faSVD)));
[~,gamkKSVD] = min(abs(1-(P_dKSVD+P_faKSVD)));

% ['gamma_k = ' num2str(gammas(gamk))];
resLP = [P_dLP(gamkLP), P_faLP(gamkLP)];
resSVD = [P_dSVD(gamkSVD), P_faSVD(gamkSVD)];
resKSVD = [P_dKSVD(gamkKSVD), P_faKSVD(gamkKSVD)];

figure;
hold on
plot(P_faLP,P_dLP,P_faSVD,P_dSVD,P_faKSVD,P_dKSVD );
legend('LP-KSVD','SVD', 'K-SVD');
tag = ['ROC for WMSC using LP-KSVD, SVD, and K-SVD', 'Asp/obs = ', num2str(sigA)];
title(tag); xlabel('P_{FA} (%)'); ylabel('P_{CC} (%)');
plot([P_faLP(gamkLP),P_dLP(gamkLP);P_faSVD(gamkSVD),P_dSVD(gamkSVD);P_faKSVD(gamkKSVD),P_dKSVD(gamkKSVD)],'o');
axis([0, 1, 0, 1]);
hold off

% adjusting our decisions to 'Previously computed kneepoint gamma'
dLP = zeros(K,1);
dSVD = zeros(K,1);
dKSVD = zeros(K,1);

alpha = 0.02;

[~, gamnpLP] = min(abs(alpha-P_faLP));
[~, gamnpSVD] = min(abs(alpha-P_faSVD));
[~, gamnpKSVD] = min(abs(alpha-P_faKSVD));
%['gamma_np = ' num2str(gammas(gamnp))]

 for k = 1:K
 dLP(k) = jmin_TLP(k)/jmin_NTLP(k) < gammas(gamkLP); %gammas(gamk); %If the J_m that lies closest to it's subspace is still too great
 dSVD(k) = jmin_TSVD(k)/jmin_NTSVD(k) < gammas(gamkSVD);
 dKSVD(k) = jmin_TKSVD(k)/jmin_NTKSVD(k) < gammas(gamkKSVD);
 end
 
m_YLP = dLP;
m_YSVD = dSVD;
m_YKSVD = dKSVD;

[CLP, order] = confusionmat(t_Y,m_YLP);
[CSVD, order] = confusionmat(t_Y,m_YSVD);
[CKSVD, order] = confusionmat(t_Y,m_YKSVD);
% Normalizing Confusion Matrix
for j = 1:length(order)
    CLP(j,:) = CLP(j,:)./sum(CLP(j,:)+~any(CLP(j,:)));
    CSVD(j,:) = CSVD(j,:)./sum(CSVD(j,:)+~any(CSVD(j,:)));
    CKSVD(j,:) = CKSVD(j,:)./sum(CKSVD(j,:)+~any(CKSVD(j,:)));
end
%figure; surf(C);
disp('LP Binary Confusion Matrix:');
disp(CLP);
disp('SVD Binary Confusion Matrix:');
disp(CSVD);
disp('K-SVD Binary Confusion Matrix:');
disp(CKSVD);

[CmLP, order] = confusionmat(origT,origMLP);
[CmSVD, order] = confusionmat(origT,origMSVD);
[CmKSVD, order] = confusionmat(origT,origMKSVD);
% Normalizing Confusion Matrix
for j = 1:length(order)
    CmLP(j,:) = CmLP(j,:)./sum(CmLP(j,:)+~any(CmLP(j,:)));
    CmSVD(j,:) = CmSVD(j,:)./sum(CmSVD(j,:)+~any(CmSVD(j,:)));
    CmKSVD(j,:) = CmKSVD(j,:)./sum(CmKSVD(j,:)+~any(CmKSVD(j,:)));
end
%figure; surf(C);
disp('LP M-Class Confusion Matrix:');
disp(CmLP);
disp('SVD M-Class Confusion Matrix:');
disp(CmSVD);

disp('K-SVD M-Class Confusion Matrix:');
disp(CmKSVD);


resLP
resSVD
resKSVD
