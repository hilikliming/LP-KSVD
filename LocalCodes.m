function [ X ] = LocalCodes(D, Y, tau,eta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
X = zeros(size(Y,2),size(D,2));
for j = 1:size(Y,2)
    [ws, omegaTau] = getNeighbors(Y(:,j),tau);
    G = (omegaTau-Y(:,j)*ones(1,size(omegaTau,2))'*(omegaTau-Y(:,j)*ones(1,size(omegaTau,2));
    % Updating local codes
    xhat = (G+eta*eye(size(used)))^(-1)*ones(size(used),1);
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
Dl = D;
ws = zeros(tau,1);
%Taking minimum distance dictionary elements
for i = 1:tau
    [~, ws(i)] = min(dist);
    Dl = Dl(:,dist~=ws(i)); 
end

omegaTau = D(:,ws);
end