% Coded by:     Peter Sutor Jr.
% Last edit:    5/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Solves the Sparse Inverse Covariance Selection problem. Sparse Inverse
% Covariance Selection minimizes for x the objective function:
%    obj(X) = trace(S*X) - log(det(X)) + lambda*||X||_1, 
% where S = covariance(D) is a square matrix of the covariance of D. Thus,
% X is a square matrix as well. The parameter lambda here is the l_1
% regularization parameter.
%
% In more detail, suppose we have a dataset D of samples from a Gaussian
% distribution in R^n, with zero mean. Let a single row of D, of length n,
% be a single sample. Then, for each row i, D_i ~ Normal(0, Sigma), 
% i = 1, ..., N, where Sigma is a matrix of covariances. Suppose we wish to
% estimate Sigma under the assumption that its inverse is sparse. Note that
% any entry in the inverse of Sigma is 0 if and only if the corresponding
% components of the random variable are conditionally independent, given
% the other variables. Because of this, this problem is similar to the
% structure learning problem of estimating the topology of the undirected
% graph of the Gaussian. Determining the sparsity pattern of the inverse of
% Sigma is known as the Covariance Selection problem. Thus, given the
% sparsity, and an empirical covariance of a dataset D, called S, we can
% formulate this problem as minimizing the loss function:
%    l(X) = trace(S*X) + log(det(X))
% The l_1 regularization version of this problem is the obj function above.
% Thus, this problem is referred to as Sparse Inverse Covariance Selection.
% 
% The ADMM Augmented Lagrangian here is:
%    L_rho(X,Z,U) = f(X) + g(Z) + (rho/2)(||X - Z + U||_2)^2 + constant(U),
% where f(X) = l(X) and g(Z) = lambda*||Z||_1 is the regularization term. 
% The rho in L_rho is the dual step size parameter. ADMM solves the problem
% of minimizing: f(X) + g(Z), subject to X - Z = 0.
%
% ADMM requires proximal operators for minimizing L_rho(X,Z,U) for X and Z,
% separately - i.e., proximal operators for functions f and g. The proximal
% operators are obtained as follows:
% 
% For the f minimization, observe that the gradient of the Augmented
% Lagrangian requires:
%    grad_X(L_rho(X,Z,U)) = grad_X(f(X)) + rho*(X - Z + U)
%                         = S - 1/X + rho*(X - Z + U) := 0
% Thus, we have the condition:
%    rho*X - 1/X := rho*(Z - U) - S                                     (1)
% To satisfy (1), we must construct such a matrix X from Z, U, S and rho
% only (the only prior information we have in ADMM). We also have an
% implicit condition that X is positive semi-definite. To this, we begin by
% taking the orthogonal eigenvalue decomposition of the right-hand side of
% (1):
%    rho*(Z - U) - S = Q*E*Q^T,
% where Q contains the eigenvectors and E is a diagonal vector of
% corresponding eigenvalues. Note that Q^T*Q = Q*Q^T = I. Thus, multiplying
% the left-hand side of (1) by Q^T on the left and Q on the right gives:
%    Q^T*(rho*X - 1/X)*Q = rho*(Q^T*X*Q) - 1/(Q^T*X*Q)
% and the right-hand side of (1) thus becomes:
%    Q^T*(rho*(Z - U) - S)*Q = Q^T*(Q*E*Q^T)*Q = (Q^T*Q)*E*(Q^T*Q) = E
% So, we want a matrix V = Q^T*X*Q, such that: rho*V - 1/V = E. This, of
% course, must have a diagonal solution, as E is diagonal. Using the
% Quadratic Formula, each i'th diagonal entry must satisfy:
%    V_{i,i} = 1/(2*rho)*[lambda_i + sqrt(lambda_i^2 + 4*rho)]
% where lambda_i is the eigenvalue at E_{i,i}. As rho is positive, these
% entries are also always positive. Thus, X = Q*V*Q^T satisfies (1) and is
% our optimal X.
%
% For function g, the Z-update is simply soft-thresholding performed on
% matrix X + U instead of a vector, with lambda/rho as the threshold.
%
% Check the covarianceselection section of function getproxops to see the 
% code for the proximal operators.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the covarianceselection function is executed with no inputs, it
% will run covarianceselectiontest with no inputs. This will generate a 
% random problem of size m = 2^9, n = 2^6, and test it for correctness, 
% showing the results.

function results = covarianceselection(D, lambda, options)
% INPUTS ------------------------------------------------------------------
% D:        Data matrix whose rows are individual measurements on which we
%           compute the covariance.
% lambda:   The regularization parameter lambda.
% options:  A struct containing options customizing the ADMM execution. If
%           no options are provided (an empty or irrelevant struct is given
%           as the options parameter), default settings are applied, as 
%           mentioned in the user manual. 
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

tic;                                    % Start timing.

% If no arguments are given, run a demo test by running the
% covarianceselectiontest function with no arguments. Returns test results 
% for random data of size m = 2^9, n = 2^6.
if nargin == 0
    display(['No arguments detected; running a demo test of the', ...
        ' Sparse Inverse Covariance Selection problem for random data', ...
        ' of size m = 2^9 and n = 2^6:']);
    results = covarianceselectiontest();
    results.solverruntime = toc;        % End timing.
    
    return;
end

% Check for any errors in inputs.
errorcheck(D, 'ismatrix', 'D', struct());
lambda = errorcheck(lambda, 'ispositivereal', 'lambda', struct());
errorcheck(options, 'isstruct', 'options', struct());

% Obtain the covariance of D and setup the args struct for function 
% getproxops.
S = cov(D);
args.S = S;
args.lambda = lambda;

n = size(S, 1);         % Size of square covariance matrix.

% Obtain proximal operators for this problem.
[minx, minz] = getproxops('CovarianceSelection', args);

% Set efficient options for ADMM for Basis Pursuit.
options.A = 1;          % Constant for A = I, for efficiency.
options.B = -1;         % Constant for B = -I, for efficiency.
options.c = 0;          % Constant for c = 0 vector, for efficiency.
options.m = n;          % Number of rows in constraint (must be set as 
                        % A, B and c are constants).
options.nA = n;         % Number of columns in A (must be set as A = 1).
options.nB = n;         % Number of columns in B (set since B = -1).

% Initialize the starting values of x, z, and u to be square matrices.
options.x0 = zeros(n, n);
options.z0 = options.x0;
options.u0 = options.x0;

% Set the objective function.
options.obj = @(x, z) trace(S*x) - log(det(x)) + lambda*norm(z(:), 1);

% Run ADMM on this setup and return the results.
results = admm(minx, minz, options);
results.solverruntime = toc;                % End timing.

end
% -------------------------------------------------------------------------