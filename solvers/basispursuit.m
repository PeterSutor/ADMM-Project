% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Solves the Basis Pursuit problem. Basis Pursuit minimizes for x the 
% objective function:
%    obj(x) = ||x||_1, 
%    subject to D*x = s,
% where D is a matrix and s is a column vector of appropriate length. Thus,
% x is a column vector of the same length.
% 
% The ADMM Augmented Lagrangian here is:
%    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + constant(u),
% where f(x) is the indicator function that x is not in the set 
% {x: D*x = s} and g(z) = ||z||_1. The rho in L_rho is the dual step size 
% parameter. ADMM solves the equivalent problem of minimizing: f(x) + g(z),
% subject to x - z = 0.
%
% ADMM requires proximal operators for minimizing L_rho(x,z,u) for x and z,
% separately - i.e., proximal operators for functions f and g. The proximal
% operators are obtained as follows:
% 
% Cache the following matrix: P = I - D^T*(D*D^T)^-1*D, specified by the 
% user. Likewise, cache q = D^T*(D*D^T)^-1*s. A projection of v onto the 
% set {x: D*x = s} is then x = P*v + q. Note that this collapses into 
% x = D^-1*s if D is square, but otherwise still works despite the 
% dimensions of D; this is the product of s with the Pseudoinverse of D.
% Thus, as the proximal function for f requires minimizing the indicator
% function (i.e., 0 for an x that IS in the set {x: D*x = s}), we can
% simply evaluate x = P*v + q for v = z - u.
%
% For the proximal function for g, note that g's formulation is
% minimizable via soft-thresholding, a proximal mapping technique:
%    z = min_z(||z - v||_2^2 + lambda*||v||_1) 
%      = sign(z)*(|z| - lambda/rho)_+ (non positive parts assigned 0)
% This function simply evaluates this for v = u + x, and appropriate rho.
% Also note that soft-thresholding typically sets rho = 1, but we move
% according to the step size parameter rho.
%
% Check the basispursuit section of function getproxops to see the proximal
% operators.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the basispursuit function is executed with no inputs, it will
% run basispursuittest with no inputs. This will generate a random
% problem of size m = 2^6, n = 2^7, and test it for correctness, showing
% the results.

function results = basispursuit(D, s, options)
% INPUTS ------------------------------------------------------------------
% D:        The m by n 'data' matrix D in the objective function constraint 
%           above.
% s:        The vector of length m in the objective function constraint 
%           above.
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
% basispursuittest function with no arguments. Returns test results for 
% random data of size m = 2^6, n = 2^7.
if nargin == 0
    display(['No arguments detected; running a demo test of the Basis', ...
        ' Pursuit problem for random data of size m = 2^6 and n = 2^7:']);
    results = basispursuittest();
    results.solverruntime = toc;        % End timing.
    
    return;
end

% Check for any errors in inputs and set number of columns for problem and
% altered vector s.
[n, s] = errorcheck(D, s, options);

% Cache some matrix / vector results for efficiency. Used to compute
% Pseudoinverse of D times a vector.
DDt = D*D';                 % Cached D*D^T.
Dsol = DDt \ D;             % Cached solution to D*D^Tx = D.
ssol = DDt \ s;             % Cached solution to D*D^T
P = (eye(n) - D'*Dsol);     % Pseduoinverse part of projection in x update.
q = D'*ssol;                % Cached second term of projection in x update.

% Set the arguments for Basis Pursuit problem in getproxops.
args.P = P;
args.q = q;

% Obtains proximal operators for f and g to use in ADMM for Basis Pursuit.
[minx, minz] = getproxops('BasisPursuit', args);

% Set efficient options for ADMM for Basis Pursuit.
options.A = 1;              % Constant for A = I, for efficiency.
options.B = -1;             % Constant for B = -I, for efficiency.
options.c = 0;              % Constant for c = 0 vector, for efficiency.
options.m = n;              % Number of rows in constraint (must be set as 
                            % A, B and c are constants).
options.nA = n;             % Number of columns in A (must be set as A = 1).
options.nB = n;             % Number of columns in B (set since B = -1).
options.solver = 'basispursuit';

% Set the objective function (z is unused).
options.obj = @(x, z) norm(x, 1);

% Run ADMM on this setup and return the results.
results = admm(minx, minz, options);
results.solverruntime = toc;                % End timing.

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Simply checks for any user error on same input, stops if error is
% detected. If input is correct, return info needed for problem.

function [n, s] = errorcheck(D, s, options)
% INPUTS ------------------------------------------------------------------
% Same as for the basispursuit function.
% 
% OUTPUTS -----------------------------------------------------------------
% n:    The column size n for m by n D matrix.
% s:    The column vector s in problem specification.
% -------------------------------------------------------------------------


% Check that D is a matrix and get number of rows.
if ~ismatrix(D)
    % Case that we weren't given a matrix. Report error and terminate.
    error('Argument D is not a matrix!');
else
    % Obtain the size of D.
    [mD, nD] = size(D);
end

% Check that s is a vector and get its size.
if ~isvector(s)
    % Case that we weren't given a valid vector. Report error and
    % terminate.
    error('Argument s is not a vector!');
else
    if isrow(s)     % If not a column vector, make it one.
        s = s';
    end
    
    % Obtain the number of rows in vector s.
    [ms, ~] = size(s);
end

% Check if dimensions match up on input vectors, terminating and reporting
% errors otherwise.
if (mD == nD && mD == ms)
    error(['Square matrix problem Dx = s; ', ...
        'don''t need Basis Pursuit to solve this!']);
elseif (mD > nD && mD == ms)
    error(['Overdetermined system Dx = s, as D has more rows than', ...
        'columns; use Unwrapped ADMM solver for efficiency, instead.']);
elseif mD ~= ms
    error(['The number of rows in matrix D must match the number of ', ...
        'rows in signal vector s!']);
elseif mD < nD
    n = nD;         % Set the column size of the problem.
end

% Check that the given options is actually a struct...
if ~isstruct(options)
    error('Given options is not a struct! At least pass empty struct!');
end

end
% -------------------------------------------------------------------------