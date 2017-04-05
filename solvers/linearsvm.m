% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Trains Linear Support Vectors on a given data matrix D, returning 
% dividing hyperplane as optimal x parameter. ADMM minimizes for x the 
% objective function:
%    obj(x) = 1/2*||x||_2^2 + C*hinge(D*x)
% where hinge(v) = sum_i{max(1 - ell_i*v_i, 0)} is the hinge loss function,
% D is vector/matrix of data (each row is a data point/vector) on which we
% train the SVM, and C is a regularization parameter that specifies whether
% we use hard linear classification (C = 0) or soft (and to what degree)
% classification (C > 0), where we allow some points to pass the dividing
% hyperplane from either class. The vector ell is a vector of 
% classification labels for our classifier; we use consecutive nonnegative 
% integers as classification labels, i.e., 0,1,...,n-1, where n is the 
% number of classes.
%
% The ADMM Augmented Lagrangian here is:
%    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||D*x - z + u||_2)^2 + 
%                   constant(u),
% where f(x) = 1/2||x||_2^2 and g(z) = C*hinge(z). The rho in L_rho is the
% dual step size parameter. ADMM solves the equivalent problem of 
% minimizing: f(x) + g(z), subject to D*x - z = 0. Alternatively, Unwrapped
% ADMM with Transpose Reduction allows the use of the zero-one loss 
% function for g(z), which indicates if z would have been classified 
% incorrectly (thus correct classification is minimizer).
%
% To solve this problem, we need to supply ADMM with proximal operators for
% f and g. We can determine these as follows:
%
% For the proximal operator for f, we note that we want to minimize
% L_rho(x,z,u) for x; as f is differentiable for x, L_rho is as well. We
% simply set the gradient of L_rho equal to 0 and solve for x to determine
% the minimizing value:
%    grad_x(L_rho(x,z,u)) := 0
%       <--> grad_x(f(x)) + rho*D^T*(D*x - z + u) = 0
%       <--> x + rho*D^T*(D*x - z + u) = 0
%       <--> (I + rho*D^T*D)*x = rho*D^T*(z - u)                        (1)
%       <--> x = (I + rho*D^T*D)^-1*[rho*D^T*(z - u)]
% Thus, we just solve the system in (1) to obtain our minimized x, and this
% is our proximal operator.
% 
% For the proximal operator for g, we recognize that the hinge loss is 
% piecewise differentiable. Thus, we proceed as for function f:
%    grad_z(L_rho(x,z,u)) := 0
%       <--> 0 = grad_z(g(z)) - rho*(D*x - z + u)
%       <--> 0 = 1/rho*grad_z(g(z)) - D*x + z - u
%       <--> z = (D*x + u) - 1/rho*grad_z(g(z))                         (2)
% Now note that: 
%    grad_z(g(z)) = grad_z(C*sum_i{max(1 - ell_i*z_i, 0)})
%                 = C*sum_i{max(grad_z(1 - ell_i*z_i), 0)}
%                 = C*sum_i{max(-ell_i*min(1 - ell_i*z_i, 1), 0)}
% Plugging this into (2) gives:
% z = (D*x + u) - C/rho*sum_i{max(-ell_i*min(1 - ell_i*z_i, 1), 0)}
%   = (D*x + u) + ell^T*max(min(1 - ell^T*z, C/rho), 0)
% This is our proximal operator.
%
% Suppose we now want to use the zero-one loss function instead of the 
% hinge function. Normally this is not possible, but thanks to Unwrapped 
% ADMM and Transpose Reduction, this is possible. We use the Pseudoinverse
% of D to do this, D^+ = (D^T*D)^-1*D^T, which is formed by solving a 
% linear system and is cached. Then the x update from before just becomes
% D^+*(z - u), our new and improved proiximal operator for f. For g, the 
% zero-one loss proximal operator seeks to argmin zo(z) + 
% rho/(2*C)||z - v||^2 for z, and zo is the zero-one function, the 
% indicator that z is wrongly classified (minimized when z is correctly 
% classified). Clearly then, zo minimized is whenever v's entries are 
% correctly classified or when v_i < 1 - sqrt(2*C/rho) (this is because
% Transpose Reduction makes rows independent of each other, thus we can 
% solve simple inequalities to arrive at these conclusions).
%
% You can set whether to use the hinge or 0-1 loss function in the options
% argument by options.lossfunction = ('hinge' or '01'). By default, 
% 'hinge' loss function is used (for either incorrect input for 
% options.lossfunction or if the field does not exist).
%
% Check the linearsvm section of function getproxops to see the proximal
% operators.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the linearsvm function is executed with no inputs, it will run
% linearsvmtest with no inputs. This will generate a random problem of size
% m = 2^8, n = 2, with two classes labelled (+1, -1), where +1 means a
% point x is above the x_1 = x_2 line, and -1 below. The data is generated
% to be linearly separable in this way. For label +1, 2^7 random points are
% generated, and the same for label -1.
        
function results = linearsvm(D, ell, C, options)
% INPUTS ------------------------------------------------------------------
% D         The data matrix to train on (matrix for x variable).
% ell       Labels for rows in D.
% C         Regularization parameter.
% hinge     If 1, use Hinge loss function, if 0 use 0-1 loss function.
% 
% OUTPUTS -----------------------------------------------------------------
% results:  A struct containing the results of the execution, including the
%           optimized values for x, z and u that optimize the objective for
%           x, runtime evaluations, records of each iteration, etc. Consult
%           the user manual for more details on the results struct, or
%           simply check what the variable contains in the Matlab 
%           interpreter after execution.
% -------------------------------------------------------------------------


% Persistent global variable to indicate whether paths have been set or
% not.
global setup;

% Check if paths need to be set up.
if isempty(setup)
    currpath = pwd;                         % Save current directory.
    
    % Get directory of this function (assumed to be in solvers folder of 
    % ADMM library).
    filepath = mfilename('fullpath');       % Get current file path.
    filepath = filepath(1:length(filepath) - length(mfilename()));
    
    % Switch to directory containing setuppaths and run it. Then switch
    % back to original directory. Save setup = 1 to indicate to all other
    % functions that setup has already been done.
    cd(filepath);
    cd('..');
    setuppaths(1);
    cd(currpath);
    setup = 1;
end

tic;                                            % Start timing.

% If no arguments are given, run a demo test by running the linearsvmtest
% function with no arguments. Creates a test as described in the NOTE above
% in the description.
if nargin == 0
    display(['No arguments detected; running a demo test of the', ...
        ' Linear SVM problem for random data of size m = 2^8, n = 2,',...
        ' for 2 classes with labels +1 and -1; for data point x,', ...
        ' +1 means it is above the x_1 = x_2 axis, -1 means below:']);
    results = linearsvmtest();
    results.solverruntime = toc;                % End timing.
    
    return;
end

% Check input for errors and return fixed values to account for weird (but
% fixable) input.
[D, ell, C] = errorchecker(D, ell, C, options);

% Check if the user wanted to use a different loss function from the hinge
% loss function (i.e., 0-1 loss function).
if (isfield(options, 'lossfunction'))
    loss = options.lossfunction;
else
    loss = 'hinge';
end

% % Obtain the chosen value for rho by user, or choose 1 by default.
% if (isfield(options, 'rho'))
%     rho = options.rho;
% else
%     rho = 1.0;
% end
% 
% [m, n] = size(D);                               % Size of D.

% Decide whether to use parallel ADMM from user input.
if isfield(options, 'parallel')
    if strcmp(options.parallel, 'both') || ...
        strcmp(options.parallel, 'zming') || ...
        strcmp(options.parallel, 'xminf')
        options.parallel = 'both';
        parallel = 1;
    else
        parallel = 0;
    end
else
    parallel = 0;
end

if ~parallel
    % Create Pseudoinverse described above.
    Dplus = pinv(D);                                % Pseudo-inverse of D.
    args.Dplus = Dplus;                             % Populate args.Dplus.
else
    % Obtain user given slices, or set to 0 to indicate distributed
    % workload.
    if isfield(options, 'slices')
        slices = options.slices;
    else
        slices = 0;
    end
    
    % Set up parallel pool...
    pool = gcp;
    
    % Process the slices argument.
    sliceopts.slicelength = size(D, 1);
    sliceopts.workers = pool.NumWorkers;
    slices = errorcheck(slices, 'slices', 'options.slices', sliceopts);
    
    % Set slices for args struct.
    args.slices = slices;
end

% Populate the args struct with info needed to get proximal operators for
% Linear SVMs.
args.D = D;
args.Dt = D';
args.ell = ell;
args.C = C;
args.lossfunction = loss;

% Obtain the proximal operators.
[~, minz] = getproxops('LinearSVM', args);

% % Set options struct for ADMM (with efficient constants for simple
% % operations on constraints).
% options.A = D;
% options.At = D';
% options.B = -1;
% options.mB = m;
% options.nB = m;
% options.c = 0;
% options.m = m;
% options.rho = rho;

% Decide which objective function to use.
if strcmp(loss, 'hinge')                        % Hinge loss objective.
    options.obj = @(x, z) 1/2*norm(x, 'fro')^2 + ...
        C*sum(max(1 - ell.*(D*x),0));
else                                            % 0-1 loss objective.
    options.obj = @(x, z) 1/2*norm(x, 'fro')^2 + ...
        C*sum(max(sign(1 - ell.*(D*x)),0));
end

% % Run ADMM and obtain our support vector x.
% results = admm(minx, minz, options);
% results.solverruntime = toc;    % End timing.
results = unwrappedadmm(minz, D, options);
results.solverruntime = toc;


end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Simply checks for any user error on same input, stops if error is
% detected. If input is correct, return info needed for problem.

function [D, ell, C] = errorchecker(D, ell, C, options)
% INPUTS ------------------------------------------------------------------
% Same as for the linearsvm function.
% 
% OUTPUTS -----------------------------------------------------------------
% D:    The data matrix D to train on in the linear SVM problem.
% ell:  The vector of classification labels we classify with.
% C:    The regularization parameter. If C = 0, we use hard linear
%       classification, else if C > 0, we use soft linear classification.
% -------------------------------------------------------------------------


% Check to make sure C is a valid value.
if (~isnumeric(C) || real(C) < 0 || ~isequal(size(C), [1 1]))
    error('Given regularization parameter C is not a nonnegative number!');
else
    C = real(C);                % Assign C.
end

% Check that ell is a vector and get its size.
if ~isvector(ell)
    error('Argument ell is not a vector!');
elseif ~isnumeric(ell)
    error('Argument ell is not numeric!');
else
    if isrow(ell)              % If not a column vector, make it one.
        ell = ell';
    end
    
    % Get size of vector ell.
    [m, ~] = size(ell);
end

% Check that D is a matrix, and that the product ell*D is valid.
if ~ismatrix(D)
    error('Given argument D is not a matrix!');
elseif ~isnumeric(D)
    error('Given matrix D is not numeric!');
else
    [mD, ~] = size(D);          % Obtain number of rows in D.
    
    if (m ~= mD)
        error('Product ell*D is not possible; sizes incompatible!');
    end
end

% Check that the given options is actually a struct...
if ~isstruct(options)
    error('Given options is not a struct! At least pass empty struct!');
end

end
% -------------------------------------------------------------------------