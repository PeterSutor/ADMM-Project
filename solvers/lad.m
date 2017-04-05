% Coded by:     Peter Sutor Jr.
% Last edit:    5/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Solves the Least Absolute Deviations (LAD) problem with ADMM. LAD
% minimizes for x the objective function:
%    obj(x) = ||D*x - s||_1, 
% where D is a data matrix and s is a signal vector. We can rephrase this
% in ADMM form as the following objective function to minimize:
%    obj(x,z) = ||z||_1,
%    subject to D*x - z = s
% Here, f(x) = 0, and g(z) = ||z||_1. The ADMM Augmented Lagrangian for
% this problem is:
%    L_rho(x,z,u) = g(z) + (rho/2)(||D*x - z - s + u||_2)^2 + constant(u),
% The rho in L_rho is the dual step size parameter. ADMM solves the problem
% of minimizing function obj(x,z) under the constraint.
%
% ADMM requires proximal operators for minimizing L_rho(x,z,u) for x and z,
% separately - i.e., proximal operators for functions f and g. The proximal
% operators are obtained as follows:
% 
% For function f, we minimize by taking the gradient for x and setting it
% equal to 0, then solving for x:
%    grad_x(L_rho(x,z,u)) = D^T*(D*x - z - s + u) := 0
% This implies that solving the following system for x gives our minimizing
% x:
%    D^T*D*x = D^T*(z + s - u)                                          (1)
% To solve this efficiently, we do not compute (D^T*D)^{-1} directly, but
% instead find its Cholesky decomposition, D^T*D = R*R^T, where R and R^T
% are lower and upper triangular, respectively. Then, (1) can be solved by
% first solving the system R*y = D^T*(z + s - u) for y, and then solving
% the system R^Tx = y for x. We cache the factors R and R^T and use system
% solving backslash operator A \ b to solve a linear system A*x = b
% efficiently.
%
% For function g, we minimize by soft-thresholding, as this is a l_1
% regularization problem. The soft-thresholding parameters here are
% v = D*x - s + u and t = 1/rho.
%
% Check the lad section of function getproxops to see the code for the
% Least Absolute Deviations (LAD) proximal operators.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the lad function is executed with no inputs, it will run ladtest
% with no inputs. This will generate a random problem of size m = 2^9,
% n = 2^6, and test it for correctness, showing the results.

function results = lad(D, s, options)
% INPUTS ------------------------------------------------------------------
% D:        Data matrix whose rows are individual measurements on which we
%           compute the covariance.
% s:        Noisy signal vector s in the problem above.
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

% If no arguments are given, run a demo test by running the ladtest 
% function with no arguments. Returns test results for random data of size
% m = 2^9, n = 2^6.
if nargin == 0
    display(['No arguments detected; running a demo test of the', ...
        ' Least Absolute Deviations problem for random data', ...
        ' of size m = 2^9 and n = 2^6:']);
    results = ladtest();
    results.solverruntime = toc;        % End timing.
    
    return;
end

% Check for any errors in inputs.
errorcheck(D, 'ismatrix', 'D', struct());
errorcheck(s, 'iscolumnvector', 's', struct());
errorcheck(options, 'isstruct', 'options', struct());

% Obtain sizes of D and s.
[m, n] = size(D);
[ms, ~] = size(s);

% Make sure that dimensions of D and s are correct.
if (m ~= ms)
    error('The number of rows in argument D do not match size of s!');
end

% Determine whether the user is going to use relaxation or not.
if (isfield(options, 'relax') && options.relax ~= 1)
    args.userelax = 1;
end

% Set the arguments struct for getproxops.
args.D = D;             % Pass matrix D.
args.s = s;             % Pass signal vector s.

% Lower triangular Cholesky factor of D^T*D. Used to efficiently compute x
% in proximal operator.
args.R = chol(D'*D, 'lower');

% Obtain the proximal operators for this problem.
[minx, minz] = getproxops('lad', args);

% Set the options struct to necessary and efficient values.
options.A = D;          % Constraint matrix A = D in this problem.
options.B = -1;         % Constraint matrix B = -I; -1 for efficiency.
options.c = s;          % Constraint vector c = s, the signal vector.
options.m = m;          % Length of vector s.
options.nA = n;         % Length of solution vector x.
options.nB = m;         % Length of vector z.

% Set the objective function.
options.obj = @(x, z) norm(z, 1);

% Run ADMM on this setup and return the results.
results = admm(minx, minz, options);
results.solverruntime = toc;                % End timing.

end
% -------------------------------------------------------------------------