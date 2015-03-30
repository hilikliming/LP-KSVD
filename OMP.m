function X = OMP(D, Y, L, errorGoal)
%******************************************************************************
% Sparse coding of a group of signals based on a given dictionary and specified
% number of atoms to use.

% input arguments:
% D: the dictionary (its columns MUST be normalized).
% Y: the signals to represent
% L: the max. number of coefficients for each signal. Empty if error 
%    used as goal
% errorGoal: the maximal allowed representation error for each signal. Empty if
%   # coefficients used as goal.

% output arguments:
% X: sparse coefficient matrix.
%******************************************************************************

X = sparse( zeros(size(D, 2), size(Y, 2)) );

for k = 1:size(X, 2) %number of vectors to operate on
    
    select_ind = zeros(size(D, 2), 1);
    % we'll allocate enough space for maximum possible iterations so we don't
    % have to concatenate, but only use as many rows as neccessary
    Q_D_t = zeros(size(D'));

    r_t = Y(:, k);
    for iter = 1:L %sparsity required. L nonzero terms
        [r_t, Q_D_t, X(:, k), select_ind] ...
            = matchAtom(r_t, Q_D_t, X(:, k), select_ind, D, Y(:, k), iter);
        
        if (norm(r_t) < errorGoal)
            break
        end
    end
end


%******************************************************************************
% Selects the next atom used to represent an observation, calculates the 
% corresponding coefficient, and updates all the variables needed to perform
% this process recursively. See paper for definitions of inputs and outputs.
%******************************************************************************
function [r_t, Q_D_t, x_t, select_ind] ...
    = matchAtom(r_t, Q_D_t, x_t, select_ind, D, y, iter)

prev_range = 1:(iter-1);

% select atom to use
proj = r_t' * D;
[~, sort_ind] = sort(abs(proj), 'descend');
new_ind = sort_ind(1);

% make sure we are selecting a new atom (round off error can cause issues)
cur_ind = 2;
while sum(ismember(select_ind, new_ind))
    new_ind = sort_ind(cur_ind);
    cur_ind = cur_ind + 1;
    %disp('Repeat atom selected!')
end
d_k_t = D(:, new_ind);

% update recursion variables
b_tm1 = Q_D_t(prev_range, :) * d_k_t;
if (iter == 1)
    % no atoms selected yet (initialization)
    d_tilde = d_k_t;
    q_t = d_tilde / (d_tilde' * d_tilde);
    Q_D_t(iter, :) = d_k_t' / (d_k_t' * d_k_t);
else
    d_tilde = d_k_t - (D(:, select_ind(prev_range)) * b_tm1);
    q_t = d_tilde / (d_tilde' * d_tilde);
    Q_D_t(prev_range, :) = Q_D_t(prev_range, :) - (b_tm1 * q_t');
    Q_D_t(iter, :) = q_t';
end

% update coefficient vector
alpha_t = q_t' * y;
x_t(select_ind(prev_range)) = x_t(select_ind(prev_range)) - (alpha_t * b_tm1);
select_ind(iter) = new_ind;
x_t(new_ind) = alpha_t;

% update residual
r_t = r_t - (alpha_t * d_tilde);
