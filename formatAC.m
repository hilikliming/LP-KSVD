function [ TS ] = formatAC( AC, N_s )
%% Input:
%  AC - columns are aspects
%  N_s - Desired frequency bin number per aspect observation
%% Output:
%  TS - Target Strength AC with N_s frequency bins
AC = 20*log10(AC);
TS = zeros(N_s,size(AC,2));
for k = 1:size(AC,2)
TS(:,k) = resample(AC(:,k),N_s,size(AC,1));
end

end

