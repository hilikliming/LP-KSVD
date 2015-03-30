function [Dictionary, output] = LPKSVD(Data, param, coding)
% =========================================================================
%%                   Locality Preserving K-SVD algorithm
% =========================================================================
% The K-SVD algorithm finds a dictionary for linear representation of
% signals. Given a set of signals, it searches for the best dictionary that
% can sparsely represent each signal. Detailed discussion on the algorithm
% and possible applications can be found in "The K-SVD: An Algorithm for 
% Designing of Overcomplete Dictionaries for Sparse Representation", written
% by M. Aharon, M. Elad, and A.M. Bruckstein and appeared in the IEEE Trans. 
% On Signal Processing, Vol. 54, no. 11, pp. 4311-4322, November 2006. 
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
    
    % *** find the coefficients using sparse coding ***
    if strcmp(coding.method, 'MP')
        if ~coding.errorFlag
            CoefMatrix = OMP([FixedDictionaryElement,Dictionary],Data, coding.L, coding.errorGoal);
        else
            CoefMatrix = OMP([FixedDictionaryElement,Dictionary],Data, coding.L, coding.errorGoal);
        end
        
    elseif strcmp(coding.method, 'BP')
        if ~coding.errorFlag
            CoefMatrix = basisPursuit([FixedDictionaryElement, Dictionary], Data, 'exact',...
                [], [], false);
        else
            CoefMatrix = basisPursuit([FixedDictionaryElement, Dictionary], Data, 'denoise',...
                coding.errorGoal, coding.denoise_gamma, false);
        end
    end
    

    % *** update dictionary one atom at a time (using function I_findBetterDictionaryElement) ***
    replacedVectorCounter = 0;
	rPerm = randperm(size(Dictionary,2));
    for j = rPerm
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(Data,...
            [FixedDictionaryElement,Dictionary],j+size(FixedDictionaryElement,2), CoefMatrix);
        Dictionary(:,j) = betterDictionaryElement;
        if (param.preserveDCAtom)
            tmpCoef = FixedDictionaryElement\betterDictionaryElement;
            Dictionary(:,j) = betterDictionaryElement - FixedDictionaryElement*tmpCoef;
            Dictionary(:,j) = Dictionary(:,j)./sqrt(Dictionary(:,j)'*Dictionary(:,j));
        end
        replacedVectorCounter = replacedVectorCounter+addedNewVector;
    end

    
    % *** display progress in terms of alternate constraint (# coefficients or error) ***
    if param.displayProgress
        output.totalerr(iterNum) = sqrt(sum(sum((Data-[FixedDictionaryElement Dictionary]*CoefMatrix).^2)) / size(Data, 2));
        output.numCoef(iterNum) = length(find(abs(CoefMatrix) >= param.coeffCutoff)) / size(Data,2);
        max_IP = max(max(Dictionary'*Dictionary - eye(size(Dictionary, 2))));
        percent_obs = 100*min(sum(abs(CoefMatrix) >= param.coeffCutoff, 2) / size(CoefMatrix, 2));
        disp(['Iter: ' num2str(iterNum) '  Avg. err: ' num2str(output.totalerr(iterNum)) ...
            '  Avg. # coeff: ' num2str(output.numCoef(iterNum)) '  Max IP: ' num2str(max_IP)...
            '  Min % obs: ' num2str(percent_obs) '%']);
    end
    
    if displayErrorWithTrueDictionary 
        [ratio(iterNum+1),ErrorBetweenDictionaries(iterNum+1)] = I_findDistanseBetweenDictionaries(param.TrueDictionary,Dictionary);
        disp(strcat(['Iteration: ', num2str(iterNum),' ratio of restored elements: ',num2str(ratio(iterNum+1))]));
        output.ratio = ratio;
    end
    
    
    % *** condition dictionary (remove redundencies, etc.) ***
    Dictionary = I_clearDictionary(Dictionary, CoefMatrix(size(FixedDictionaryElement,2)+1:end,:), Data, param);
    
    if isfield(param,'waitBarHandle')
        waitbar(iterNum/param.counterForWaitBar);
    end
    
end % end LP-KSVD loop


% *** remove atoms that are not as useful ***
for jj = size(Dictionary,2):-1:1 % run through all atoms (backwards since we may remove some)
    G = Dictionary'*Dictionary; G = G-diag(diag(G));
    if (max(G(jj,:)) > param.maxIP) ||...
            ( (length(find(abs(CoefMatrix(jj,:)) > param.coeffCutoff)) / size(CoefMatrix,2)) <= param.minFracObs )
        Dictionary(:,jj) = [];
    end
end

output.CoefMatrix = CoefMatrix;
Dictionary = [FixedDictionaryElement, Dictionary];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  findBetterDictionaryElement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Dictionary,j,CoefMatrix)

relevantDataIndices = find(CoefMatrix(j,:)); % the data indices that uses the j'th dictionary element.
if (length(relevantDataIndices)<1) %(length(relevantDataIndices)==0)
    ErrorMat = Data-Dictionary*CoefMatrix;
    ErrorNormVec = sum(ErrorMat.^2);
    [~,i] = max(ErrorNormVec);
    betterDictionaryElement = Data(:,i);%ErrorMat(:,i); %
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);
    betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));
    CoefMatrix(j,:) = 0;
    NewVectorAdded = 1;
    return;
end

NewVectorAdded = 0;
tmpCoefMatrix = CoefMatrix(:,relevantDataIndices); 
tmpCoefMatrix(j,:) = 0;% the coeffitients of the element we now improve are not relevant.
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); % vector of errors that we want to minimize with the new element


% better dictionary element and values of beta found using svd to approximate the matrix 'errors' 
% with a one-rank matrix.


[betterDictionaryElement,singularValue,betaVector] = svds(errors,1);
CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';% *signOfFirstElem


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  findDistanseBetweenDictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ratio,totalDistances] = I_findDistanseBetweenDictionaries(original,new)
% first, all the column in oiginal starts with positive values.
catchCounter = 0;
totalDistances = 0;
for i = 1:size(new,2)
    new(:,i) = sign(new(1,i))*new(:,i);
end
for i = 1:size(original,2)
    d = sign(original(1,i))*original(:,i);
    distances =sum ( (new-repmat(d,1,size(new,2))).^2);
    [~,index] = min(distances);
    errorOfElement = 1-abs(new(:,index)'*d);
    totalDistances = totalDistances+errorOfElement;
    catchCounter = catchCounter+(errorOfElement<0.01);
end
ratio = 100*catchCounter/size(original,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  I_clearDictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Dictionary = I_clearDictionary(Dictionary, CoefMatrix, Data, param)
% *** replace atoms that:
% 1) exceed maximum allowed inner product with another atom
% 2) is used an insufficient number of times for reconstructing observations

Er = sum((Data-Dictionary*CoefMatrix).^2,1); % error in representation

for jj = 1:size(Dictionary,2) % run through all atoms
    
    G = Dictionary'*Dictionary; G = G-diag(diag(G)); % matrix of inner products (diagonal removed)
    
    if (max(G(jj,:)) > param.maxIP) ||...
            ( (length(find(abs(CoefMatrix(jj,:)) > param.coeffCutoff)) / size(CoefMatrix,2)) <= param.minFracObs )
        [~, pos] = max(Er); % sorted indices of obseravtions with highest reconstruction errors
        
        % replace jj'th atom with normalized data vector with highest reconstruction error
        Er(pos(1)) = 0;
        Dictionary(:,jj) = Data(:,pos(1)) / norm(Data(:,pos(1)));
    end
end