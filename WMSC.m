% Weighted matched subspace classifier
function [ d_Y ] = WMSC( Y, D_s, mu_m, est,sigA )
%% Input
% Y = Observation matrix which we are performing classification upon
% D_s = Sparse Representation Dictionary created by K-SVD processing of
% overcomplete D
% A_m = struct containing sampling matrices of the known signal types
% the are present in the dictionary D_s

%% Output
% Decision Matrix d_Y for Observation Matrix Y
if(strcmp(est, 'LMedS'))
    lmed = true;
    lda = false;
else if(strcmp(est, 'MSD'))
    lmed = false;
    lda = false;
else if(strcmp(est, 'LDA'))
    lda = true; 
    lmed = false;
    else
        disp('Error, must select estimator Parameter MSD of LMedS (Simple MSD vs. Weighted MSD)');
        return
    end
    end
end

[N, K] = size(Y);
M = length(D_s);
J_km = zeros(K,M);
I = eye(N);
S_t = [D_s(1).D,D_s(2).D];
S_nt = [D_s(3).D,D_s(4).D,D_s(5).D];
W_lda = diag(norm(mean(S_t,2)-mean(S_nt,2))./(var(S_t')+var(S_nt'))');
%W_lda = W_lda./max(max(W_lda));
%surf(W_lda);
for m = 1:M
    H_m = double(D_s(m).D); % The columns corresponding to this signal class
    % Initialize weighting matrix to allow for defaulting to MSD
    W = I;
    A = H_m;
    P_A = A/(A'*A)*A';
    %wt = rank(P_A);
    if(lmed)% so we don't rebuild P_A a million times if we don't need to...
        %Theta = (H_m'*W*H_m)\(H_m')*W*Y;
        W = LMedS(Y,H_m);            
        A = sqrt(W)*H_m; % updating P_A now that we have a better W...
        P_A = A/(A'*A)*A';
    end
    if(lda)% so we don't rebuild P_A a million times if we don't need to...
        %Theta = (H_m'*W*H_m)\(H_m')*W*Y;
        W = W_lda;
    end
    for k = 1:K
        mult = floor(k/sigA);
        patch = [mult*sigA+1:mult*sigA+sigA];
        if(sigA+k>K)
            patch = patch(1:mod(k,sigA));
            if(sigA==1)patch = K;end
        end
        zk = W*(Y(:,patch)-repmat(double(mu_m(m).mu),1,length(patch)));
        J_km(k,m) = trace(zk'*(I-P_A)*zk)/trace(zk'*zk);%./trace(Z'*Z);
    end
    %warning('off','last');
end

d_Y = J_km;
end



