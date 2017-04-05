% Coded by:     Peter Sutor Jr.
% Last edit:    4/11/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Solves a simple model problem using ADMM. The model problem to solve is
% minimizing for x the objective function:
%    obj(x) = 1/2*(||P*x - r||_2)^2 + 1/2*(||Q*x - s||_2)^2,
% where P is a m by n matrix, Q is an m by n matrix, and r and s are
% column vectors of length m. Thus, x is a column vector of length n.
% 
% The ADMM Augmented Lagrangian here is:
%    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + constant(u),
% where f(x) = 1/2*(||P*x - r||_2)^2 and g(z) = 1/2*(||Q*z - s||_2)^2, i.e.
% such that obj(x) = f(x) + g(x). The rho in L_rho is the dual step size
% parameter. ADMM solves the equivalent problem of minimizing: f(x) + g(z),
% subject to x - z = 0.
%
% ADMM requires proximal operators for minimizing L_rho(x,z,u) for x and z,
% separately - i.e., proximal operators for functions f and g. The proximal
% operator for f is clearly the gradient of L_rho(x,z,u) for x, set equal
% to 0 and solved for x. The same is applicable for the proximal operator
% for g. Derivation below:
%
%    x = min_x(L_rho(x,z,u)) 
%       <--> x such that grad_x(L_rho(x,z,u)) := 0
%       <--> grad_x(f(x)) + rho*(x - z + u) = 0
%       <--> P^T*(P*x - r) + rho*(x - z + u) = 0
%       <--> (P^T*P - I)*x = P^T*r + rho*(z - u)
%       <--> x := (P^T*P - I)^-1*[P^T*r + rho*(z - u)]
%
% By similar steps: z := (Q^T*Q - I)^-1*[Q^T*s + rho*(x + u)], the only
% difference being a negative sign due to -z in the penalty term. Thus, the
% proximal operators to use in ADMM are:
%    prox_f(x,z,u,rho) = (P^T*P - I)^-1*[P^T*r + rho*(z - u)]
%    prox_g(x,z,u,rho) = (Q^T*Q - I)^-1*[Q^T*s + rho*(x + u)]
% Check the model section of function getproxops to see the proximal
% operators.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the model function is executed with no inputs, it will run
% modeltest with no inputs. This will generate a random model problem of
% size m = n = 2^7, and test it for correctness, showing the results.

function [results] = model(P, Q, r, s, options)
% INPUTS ------------------------------------------------------------------
% P:        The m by n matrix P in the objective function above.
% Q:        The m by n matrix Q in the objective function above.
% r:        The vector of length m in the objective function above.
% s:        The vector of length m in the objective function above.
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
% test:     A struct containing the details of the random test that was
%           performed, including the random data generated, how correct it
%           was, whether it reached the correct tolerance (successful test)
%           and so on. Consult the user manual for more details on the test
%           struct, or simply check what the variable contains in the 
%           Matlab interpreter after execution.
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

% If no arguments are given, run a demo test by running the modeltest
% function with no arguments. Returns test results for random data of size
% m = n = 2^7.
if nargin == 0
    display(['No arguments detected; running a demo test of the model', ...
        ' problem for random data of size m = n = 2^7:']);
    results = modeltest();
    results.solverruntime = toc;        % End timing.
    
    return;
end

% Perform error checking on input.
[~, n, r, s] = errorcheck(P, Q, r, s, options);

% Cache transposes.
Pt = P';
Qt = Q';

% Set arguments to get proximal operators for the model problem.
args.PtP = Pt*P;    % Cached value of P^T*P.
args.Ptr = Pt*r;    % Cached value of P^T*r.
args.QtQ = Qt*Q;    % Cached value of Q^T*Q.
args.Qts = Qt*s;    % Cached value of Q^T*s.
args.n = n;         % Number of columns in the problem.

% Get proximal operators.
[minx, minz] = getproxops('Model', args);

% Set constraints and objective function for ADMM.
options.A = 1;      % Matrix A in constraint is the identity.
options.B = -1;     % Matrix B in constraint is the negative identity.
options.c = 0;      % Vector c in constraint is the 0 vector.
options.m = n;      % Number of rows in constraint is m.
options.nA = n;     % Number of columns for matrix A in constraint.
options.nB = n;     % Number of columns for matrix B in constraint.

% The objective value for ADMM's iterations.
options.obj = @(x, z) 1/2*norm(P*x - r, 'fro')^2 + ...
    1/2*norm(Q*z - s, 'fro')^2;

% Perform ADMM on this setup.
results = admm(minx, minz, options);
results.solverruntime = toc;            % End timing.

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Simply checks for any user error on same input, stops if error is
% detected. If input is correct, return info needed for problem.

function [m, n, r, s] = errorcheck(P, Q, r, s, options)
% INPUTS ------------------------------------------------------------------
% Same as for the model function.
% 
% OUTPUTS -----------------------------------------------------------------
% m:    The size m for m by n P and Q matrices.
% n:    The size n for m by n P and Q matrices.
% r:    The column vector r in problem specification.
% s:    The column vector s in problem specification.
% -------------------------------------------------------------------------


% Check that P is a matrix and get its dimensions.
if ~ismatrix(P)
    error('Argument P is not a matrix!');
else
    [mP, nP] = size(P);
end

% Check that Q is a matrix and get its dimensions.
if ~ismatrix(Q)
    error('Argument Q is not a matrix!');
else
    [mQ, nQ] = size(Q);
end

% Check that r is a vector and get its size.
if ~isvector(r)
    error('Argument r is not a vector!');
else
    if isrow(r)     % If not a column vector, make it one.
        r = r';
    end
    
    [mr, ~] = size(r);
end

% Check that s is a vector and get its size.
if ~isvector(s)
    error('Argument s is not a vector!');
else
    if isrow(s)     % If not a column vector, make it one.
        s = s';
    end
    
    [ms, ~] = size(s);
end

% Check that sizes of matrix and vector inputs are correct.
if mP ~= mQ
    error('Number of rows in P do not match number of rows in Q!');
elseif nP ~= nQ
    error('Number of columns in P do not match number of columns in Q!');
elseif mP ~= mr
    error('Number of rows in P does not match length of vector r!');
elseif mQ ~= ms
    error('Number of rows in Q does not match length of vector s!');
else
    m = mP;         % Set the m to use in the solver.
    n = nP;         % Set the n to use in the solver.
end

% Check that the given options is actually a struct...
if ~isstruct(options)
    error(['Given options argument is not a struct!', ...
        ' Please check your arguments and try again.']);
end

end
% -------------------------------------------------------------------------