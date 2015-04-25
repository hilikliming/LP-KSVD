%% Weighted Matched Subspace Classifier
% Based on Procedures from: Robust Multidimensional Matched Subspace
% Classifiers Based on Weighted Least-Squares (IEEE Trans Signal Processing 
% Vol. 55 No. 3 March 2007)
% Last Update 3/4/15
% Hall, John J.

function [ d_Y ] = WMSC( Y, D_s,mu_m, est,sigA )
%% Input
% Y = Observation matrix which we are performing classification upon

% D_s = Struct containing M different Sparse Representation Dictionaries
% used to describe the M different Subspace Signal Classes

% mu_m = struct containing noise means under each of the M hypotheses
% (typically the average mirepresentation of FRM to Real ACs for my
% problem)

% est = String parameter which denotes which weighting matrix method to
% use (MSD is identity weighting and classic Matched Subspace Classifier)

% sigA = number of vectors used in a decision about Y (i.e. vectors per
% observation)

%% Output
% Decision Matrix d_Y for Observation Matrix Y

%%
if(strcmp(est, 'LMedS'))
    lmed = true;
else if(strcmp(est, 'MSD'))
    lmed = false;
    else
        disp('Error, must select estimator Parameter MSD of LMedS (Simple MSD vs. Weighted MSD)');
        return
    end
end

% Initializing
[N, K] = size(Y);
M = length(D_s);
J_km = zeros(K,M);
I = eye(N);

for m = 1:M
    H_m = double(D_s(m).D); % The columns corresponding to this signal class
    % Initialize weighting matrix to allow for defaulting to MSD
    W = I;
    A = H_m;
    % Projection matrix onto H_m
    P_A = A/(A'*A)*A';
    %wt = rank(P_A);
    if(lmed)% so we don't rebuild P_A a million times if we don't need to...
        %Theta = (H_m'*W*H_m)\(H_m')*W*Y;
        W = LMedS(Y,H_m);            
        A = sqrt(W)*H_m; % updating P_A now that we have a weighted W...
        P_A = A/(A'*A)*A';
    end
    % Running through all vector and making a decision for each
    for k = 1:K
        mult = floor(k/sigA);
        % Selecting relevant vector patch for deciding on observation k
        patch = [mult*sigA+1:mult*sigA+sigA];
        % Catch the end case 
        if(sigA+k>K)
            patch = patch(1:mod(k,sigA));
            if(sigA ==1) patch = K; end
        end
        % Form observation from relevant vectors and by removing mean under
        % hypothesis m
        zk = sqrt(W)*(Y(:,patch)-repmat(double(mu_m(m).mu),1,length(patch)));
        J_km(k,m) = trace(zk'*(I-P_A)*zk)/trace(zk'*zk);%./trace(Z'*Z);
    end
end
d_Y = J_km;
end



