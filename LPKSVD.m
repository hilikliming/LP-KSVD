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

%****************************
%% Populate Necessary Fields*
%****************************

if (~isfield(param,'displayProgress'))
    param.displayProgress = 0;
end

if (isfield(coding,'errorFlag')==0)
    coding.errorFlag = 0;
end

if (isfield(param,'TrueDictionary'))
    displayErrorWithTrueDictionary = 1;
    ErrorBetweenDictionaries = zeros(param.numIteration+1,1);
    ratio = zeros(param.numIteration+1,1);
else
    displayErrorWithTrueDictionary = 0;
	ratio = 0;
end

if (param.preserveDCAtom>0)
    FixedDictionaryElement(1:size(Data,1),1) = 1/sqrt(size(Data,1));
else
    FixedDictionaryElement = [];
end

if param.displayProgress
    if ~coding.errorFlag
        output.totalerr = zeros(param.numIteration, 1);
    else
        output.numCoef = zeros(param.numIteration, 1);
    end
end


%*******************
%% Define Dictionary *
%********************

if (size(Data,2) < param.K)
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    Dictionary = Data(:, 1:size(Data,2));
    return;
elseif (strcmp(param.InitializationMethod,'DataElements'))
    Dictionary = Data(:, randsample(size(Data, 2), param.K-param.preserveDCAtom));
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
    Dictionary = param.initialDictionary(:, 1:param.K-param.preserveDCAtom);
end

% *** reduce the components in Dictionary that are spanned by the fixed elements ***
if param.preserveDCAtom
    tmpMat = FixedDictionaryElement \ Dictionary;
    Dictionary = Dictionary - FixedDictionaryElement*tmpMat;
end

% *** normalize the dictionary ***
Dictionary = Dictionary * diag(1./sqrt(sum(Dictionary.*Dictionary)));
Dictionary = Dictionary .* repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.


%************************
%% Begin LP K-SVD algorithm *
%************************

for iterNum = 1:param.numIteration
    
    % *** find the coefficients using local reconstruction code method ***
%     if strcmp(coding.method, 'MP')
%         if ~coding.errorFlag
    CoefMatrix = LocalCodes([FixedDictionaryElement,Dictionary],Data, coding.tau, coding.eta);
    LRE = zeros(size(Dictionary,2),1); s = 0;
    for k = 1:size(Dictionary,2)
        xk = CoefMatrix(k,:);
        wk = find(xk~=0);
        Lambda_k = Y(:,wk);
        for j = 1:length(wk)
        LRE(k) = LRE(k) + norm(Lambda_k(:,j)-Dictionary(:,wk(j));
        s = s+ 1/length(wk)*U(:,1)'/norm(U(:,1))*Y(:,wk(j));
        end
        Ek = abs(Lambdra_k-Dictionary(:,wk)*xk(wk));
        [U,S,V] = svd(Ek,'econ');
        dknew = s*U(:,1);
        Dictionary(:,k) = dknew;  
    end
    disp(['Iteration: ', num2str(iterNum),' Avg. Number Coeff: ', num2str(mean(sum(CoefMatrix~=0,1))), ...
        'Average LRE: ', num2str(mean(LRE))]);
end % end LP-KSVD loop

% % *** remove atoms that are not as useful ***
% for jj = size(Dictionary,2):-1:1 % run through all atoms (backwards since we may remove some)
%     G = Dictionary'*Dictionary; G = G-diag(diag(G));
%     if (max(G(jj,:)) > param.maxIP) ||...
%             ( (length(find(abs(CoefMatrix(jj,:)) > param.coeffCutoff)) / size(CoefMatrix,2)) <= param.minFracObs )
%         Dictionary(:,jj) = [];
%     end
% end

output.CoefMatrix = CoefMatrix;
Dictionary = [FixedDictionaryElement, Dictionary];
end
