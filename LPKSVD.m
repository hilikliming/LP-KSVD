function [Dictionary, output] = LPKSVD(Data, param, coding)
% =========================================================================
%%                   Locality Preserving K-SVD algorithm
% =========================================================================
% The LP K-SVD algorithm finds a dictionary for linear representation of
% signals. Given a set of signals, it searches for the best dictionary that
% can represent each signal using dictionary elements that belong to a given 
% neighborhood on the smooth manifold that data is assumed to lie on. 
% Detailed discussion on the algorithm and possible applications can be 
% found in " Locality Preserving KSVD for Non-Linear Manifold Learning." by 
% Yin Zhou, Jinglun Gao, and Kenneth E. Barner. Acoustics, Speech and 
% Signal Processing (ICASSP), 2013 IEEE International Conference, May 2013.
% =========================================================================
%% INPUT ARGUMENTS:
% Data: nXN matrix that contins N signals (Y), each of dimension n. 

% param: structure that includes all required parameters for the K-SVD execution. Required fields:
%  - K: number of dictionary elements to train
%  - numIteration: number of iterations to perform.
%  - preserveDCAtom: if =1 then the first atom in the dictionary is set to be constant, and does not
%    ever change. This might be useful for working with natural images (in this case, only param.K-1
%    atoms are trained).
%  - InitializationMethod: method to initialize the dictionary, can be one of the following 
%    arguments: 1) 'DataElements' (initialization by the signals themselves),2) 'GivenMatrix' 
%    (initialization by a given matrix param.initialDictionary).
%  - initialDictionary (optional, see InitializationMethod): if the initialization method is 
%    'GivenMatrix', this is the matrix that will be used.
%  - TrueDictionary (optional): if specified, in each iteration the difference between this 
%    dictionary and the trained one is measured and displayed.
%  - displayProgress: if =1 progress information is displyed. If coding.errorFlag==0, the average 
%    repersentation error (RMSE) is displayed, while if coding.errorFlag==1, the average number of 
%    required coefficients for representation of each signal is displayed.

% *** parameters not defined in original script ***
%    - minFracObs: min % of observations an atom must contribute to, else it is replaced
%    - maxIP: maximum inner product allowed betweem atoms, else it is replaced

% coding: structure containing parameters related to sparse coding stage of K-SVD algorithm
%  - method: method used for sparse coding. Can either be 'MP' or 'BP' for matching pursuit and
%    basis pursuit, respectively.
%  - errorFlag: For MP: if =0, a fix number of coefficients is used for representation of each 
%    signal. If so, coding.L must be specified as the number of representing atom. If =1, arbitrary 
%    number of atoms represent each signal, until a specific representation error is reached. If so,
%    coding.errorGoal must be specified as the allowed error. For BP: if =0, then the solution must
%    be exact, otherwise BP denoising (with error tolerance) is used.
%  - L(optional, see errorFlag) maximum coefficients to use in OMP coefficient calculations.
%  - errorGoal(optional, see errorFlag): allowed representation error in representing each signal.
%  - denoise_gamma = parameter used for BP denoising that controls the tradeoff between 
%    reconstruction accuracy and sparsity in BP denoising.

% =========================================================================
%% OUTPUT ARGUMENTS:
%  Dictionary                  The extracted dictionary of size nX(param.K).
%  output                      Struct that contains information about the current run. It may include the following fields:
%    CoefMatrix                  The final coefficients matrix (it should hold that Data equals approximately Dictionary*output.CoefMatrix.
%    ratio                       If the true dictionary was defined (in
%                                synthetic experiments), this parameter holds a vector of length
%                                param.numIteration that includes the detection ratios in each
%                                iteration).
%    totalerr                    The total representation error after each
%                                iteration (defined only if
%                                param.displayProgress=1 and
%                                coding.errorFlag = 0)
%    numCoef                     A vector of length param.numIteration that
%                                include the average number of coefficients required for representation
%                                of each signal (in each iteration) (defined only if
%                                param.displayProgress=1 and
%                                coding.errorFlag = 1)
% =========================================================================
Dictionary = param.Dict;
% % *** normalize the dictionary ***
% Dictionary = Dictionary * diag(1./sqrt(sum(Dictionary.*Dictionary)));
% Dictionary = Dictionary .* repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.


%****************************
%% Begin LP K-SVD algorithm *
%****************************

for iterNum = 1:param.numIteration
    % *** find the coefficients using local reconstruction code method ***
    CoefMatrix = LocalCodes(Dictionary,Data, coding.tau, coding.eta);
    LRE = zeros(size(Dictionary,2),1); 
    % Relearning d_k's
    for k = 1:size(Dictionary,2)    
        xk = CoefMatrix(k,:); 
        wk = find(xk~=0)'; % find indices where xk is not zero (i.e. d_k selected);
        if(size(wk,1)>0) % don't change the d_k if no y's use it, might be used later...
            mod_dic = Dictionary;
            mod_dic(:,k) = []; %removing kth atom
            mod_coeffs = CoefMatrix;
            mod_coeffs(k,:) = [];
            E = Data - mod_dic*mod_coeffs;
            E_k = E(:,wk);      
            Lambda_k = Data(:,wk);
            %Ek = abs(Lambda_k-Dictionary*CoefMatrix(:,wk));
            [U,~,~] = svd(E_k,'econ');
            s = 0;
            for j = 1:length(wk)
            %LRE(k) = LRE(k) + norm(Lambda_k(:,j)-Dictionary(:,wk)*xk(wk));
                s = s+ 1/length(wk)*U(:,1)'/norm(U(:,1))*Data(:,wk(j));
                LRE(k) = LRE(k) + norm(Dictionary(:,k) - Lambda_k(:,j))^2;
            end
                dknew = s*U(:,1);
                Dictionary(:,k) = dknew; 
        else
                disp('atom swap');
                dknew=swapNeighbors(Data,Dictionary,k);
                Dictionary(:,k) = dknew;
        end
    end
    disp(['Iteration: ', num2str(iterNum),' Avg. Number Coeff: ', num2str(mean(sum(CoefMatrix~=0,1))), ...
        ' Average LRE: ', num2str(mean(LRE))]);
end % end LP-KSVD loop

output.CoefMatrix = CoefMatrix;
%Dictionary = [FixedDictionaryElement, Dictionary];
end

function [dnew]= swapNeighbors(Data,Dictionary,k)
dk = Dictionary(:,k);
dists = zeros(size(Data,2),1);
for i = 1:length(dists)
    dists(i)=norm(dk-Data(:,i));
end
[~,cn] = min(dists);
dnew = Data(:,cn);
end
