%% Weighted Matched Subspace Classifier
% Based on Procedures from: Robust Multidimensional Matched Subspace
% Classifiers Based on Weighted Least-Squares (IEEE Trans Signal Processing 
% Vol. 55 No. 3 March 2007)
% Last Update 3/4/15
% Hall, John J.

function [ d_Y ] = LocalWMSC( Y, D_s,mu_m,R_m,est,sigA,tau,eta )
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

for m=length(D_s):-1:1 %go backwards because we want to remove elements along way
    if(size(D_s(m).D,2)<2)
        D_s(m)=[];
    end
end

M = length(D_s);
J_km = ones(K,M);
I = eye(N);

for m = 1:M
    H_m = double(D_s(m).D); % The columns corresponding to this signal class
    % Initialize weighting matrix to allow for defaulting to MSD
    W = I;
    A = H_m;
    % Projection matrix onto H_m
    %P_A = A/(A'*A)*A';
    %wt = rank(P_A);
    tau_m = min([tau,size(A,2)]);
    X = LocalCodes(A,Y,tau_m,eta);
    if(lmed)% so we don't rebuild P_A a million times if we don't need to...
        %Theta = (H_m'*W*H_m)\(H_m')*W*Y;
        W = LMedS(Y,H_m);            
        A = sqrt(W)*H_m; % updating P_A now that we have a weighted W...
        %P_A = A/(A'*A)*A';
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
        zk = sqrt(W)*R_m(m).R*(Y(:,patch)-mu_m(m).mu*ones(1,length(patch)));
        ak=A*X(:,patch);
        J_km(k,m) = norm(zk-ak,'fro')^2/norm(zk,'fro')^2;%trace((zk*zk'-ek*ek'))/trace(zk'*zk);%./trace(Z'*Z);
    end
end
d_Y = J_km;

end



