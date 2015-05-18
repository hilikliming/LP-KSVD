%% Used in Conjunction with LPKSVD.m
% John J. Hall, April 2015
function [ X ] = LocalCodes(D, Y, tau,eta)

X = zeros(size(D,2),size(Y,2));
for j = 1:size(Y,2)
    [ws, omegaTau] = getNeighbors(Y(:,j),D,tau);
    G = (omegaTau-Y(:,j)*ones(1,size(omegaTau,2)))'*(omegaTau-Y(:,j)*ones(1,size(omegaTau,2)));
    % Updating local codes
    xhat = (G+eta*eye(size(G)))^(-1)*ones(size(G,2),1);
    xhat = xhat/sum(xhat);
    X(ws,j) = xhat;
end

end


% Finds tau nearest neighboring dictionary elements
function [ws, omegaTau] = getNeighbors(yj,D,tau)
dist= zeros(size(D,2),1);
for j = 1:size(D,2)
    dist(j) =  norm(D(:,j)-yj);
end
%Dl = D;
ws = zeros(tau,1);
%Taking minimum distance dictionary elements
for i = 1:tau
    [~, ws(i)] = min(dist);
    %Dl = Dl(:,find(dist~=del));
    %dist(dist==del) = inf; 
    %mod
    dist(ws(i)) = inf;
end

omegaTau = D(:,ws);
end