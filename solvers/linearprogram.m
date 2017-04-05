% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Solves the Linear Program problem. Linear Programming minimizes for x the 
% objective function:
%    obj(x) = <b,x> = b^T*x, 
%    subject to D*x = s, x >= 0
% where D is a matrix and s is a column vector of appropriate length. Thus,
% x and b are column vectors of the same length. Vector b represents a
% vector of coefficients in the linear program. Note that this formulation
% is known as the standard form for a linear program. One can use any conic
% constraint on x, not just x >= 0. In this case, one will have to provide
% the appropriate proximal function for g to minimize this in the options
% struct (options.altproxg).
% 
% The ADMM Augmented Lagrangian here is:
%    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + constant(u),
% where f(x) = b^T*x such that x is in the set {x: D*x = s} and g(z) is the
% indicator function that x is not in the non-negative orthant (R_+^n). The
% rho in L_rho is the dual step size parameter. ADMM solves the equivalent 
% problem of minimizing: f(x) + g(z), subject to x - z = 0.
%
% ADMM requires proximal operators for minimizing L_rho(x,z,u) for x and z,
% separately - i.e., proximal operators for functions f and g. The proximal
% operators are obtained as follows:
% 
% For the proximal function for f, we note that it is equal to the gradient
% of L_rho(x,z,u) set to zero:
%    grad_x(L_rho(x,z,u)) := 0 <--> grad_x(b^T*x) + rho*(x - z + u) = 0
%                              <--> b + rho*(x - z + u) = 0
%                              <--> rho*x + [b - rho*(z - u)] = 0       (1)
% We note that this can be solved trivially as a linear system. However,
% the solution from (1) is not dependent on D nor s. We could stick this
% condition into the z-update, but there's a another way. Furthermore, as D
% is not square, we can't assume that we can directly solve the D and s 
% system using D's inverse. Instead, we form a larger, square system with
% the properties we want and solve that. Namely, we want a square system 
% such that:
%    [ rho*I, F ]   [ x ]   [ b - rho*(z - u) ]   [0]
%    [   G  , H ] * [ y ] + [        q        ] = [0]                   (2)
% where, for D being an m by n matrix, F is n by m, G is m by n, H is n by
% m, x and b are vectors of length n, and y and q are vectors of length m.
% We choose y to satisfy our domain constraint on function f:
%    D*y = s <--> D*y - s = 0                                           (3)
% This coincides nicely with the right hand side of (1) and (2). Thus, we 
% choose G = D. We also need the contribution of F, H and q in the right
% hand side of (2) to be 0. For H, that can be trivial; set H = 0. For F,
% however, not so much. We notice that by using (3), we naturally make q =
% -s. Then, we don't want F's contribution to change the corresponding 0
% vector on the left hand side. So, we make a non-trivial requirement:
%    F*y = 0 <--> D^T*y = 0                                             (4)
% With no data for F, we arbitrarily choose F = D^T, for convenience in
% dimension and the ready availability of the transpose of D. So, pooling
% together (1 - 4) gives us a system to solve:
%    [ rho*I, D^T ]   [ x ]   [ b - rho*(z - u) ]    [0]
%    [   D  ,  0  ] * [ y ] + [       -s        ] =  [0]                  (5)
% The solution being (now that the problem is square and has an inverse:
%        [ x ]          ([ rho*I, D^T ])   [ rho*(z - u) - b ]
%    v = [ y ] = inverse([   D  ,  0  ]) * [        s        ]
% Our minimized x in the proximal function is now the x-part in solution
% vector v.
%
% For the proximal function for g, our proximal operator is simple. To
% project a given vector v = x + u into the non-negative orthant, we simply
% take the positive parts of v and set the rest to 0; this is our minimized
% vector z.
%
% Check the linearprogram section of function getproxops to see the 
% proximal operators.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the linearprogram function is executed with no inputs, it will
% run linearprogramtest with no inputs. This will generate a random
% problem of size m = 2^6, n = 2^7, and test it for correctness, showing 
% the results.

function results = linearprogram(b, D, s, options)
% INPUTS ------------------------------------------------------------------
% b:        The coefficient matrix of length n in the objective function
%           contraint above.
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

tic;                                            % Start timing.

% If no arguments are given, run a demo test by running the
% linearprogramtest function with no arguments. Returns test results for 
% random data of size m = 2^6, n = 2^7.
if nargin == 0
    display(['No arguments detected; running a demo test of the', ...
        ' Linear Progam problem for random data of size m = 2^6 and',...
        ' n = 2^7:']);
    results = linearprogramtest();
    results.solverruntime = toc;                % End timing.
    
    return;
end

% Run a check for errors on inputs, returning potentially fixed outputs.
[m, n, b, s] = errorcheck(b, D, s, options);

% Set up data for getproxops to return proximal functions.
Dt = D';                % Cache transpose of data D.
In = eye(n);            % Cache properly sized identity matrix.
zero = zeros(m);        % Cache properly sized zero matrix.

% Populate the data for the args struct to pass to getproxops.
args.D = D;
args.Dt = Dt;
args.In = In;
args.zero = zero;
args.b = b;
args.s = s;
args.n = n;

% If the user provided an alternate proximal function for g (perhaps they
% have a different conic constraint than x >= 0 for the problem), use that
% proximal function. Otherwise, use the default.
if (isfield(options, 'altproxg') && ...
    isa(options.altproxg, 'function_handle'))

    % Set proximal operators.
    [minx, ~] = getproxops('LinearProgram', args);
    minz = options.altproxg;
else
    % Get proximal operators.
    [minx, minz] = getproxops('LinearProgram', args);
end

% Set constraints and objective function for ADMM.
options.A = 1;
options.B = -1;
options.c = 0;
options.m = n;
options.nA = n;
options.nB = n;
options.obj = @(x, z) b'*x;             % The objective function.

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

function [m, n, b, s] = errorcheck(b, D, s, options)
% INPUTS ------------------------------------------------------------------
% Same as for the linearprogram function.
% 
% OUTPUTS -----------------------------------------------------------------
% m:    The row size m for m by n D matrix.
% n:    The column size n for m by n D matrix.
% b:    The column vector b in problem specification.
% s:    The column vector s in problem specification.
% -------------------------------------------------------------------------


% Check that D is a matrix and get its dimensions.
if ~ismatrix(D)
    error('Argument D is not a matrix!');
else
    [mD, nD] = size(D);
end

% Check that b is a vector and get its size.
if ~isvector(b)
    error('Argument b is not a vector!');
else
    if isrow(b)     % If not a column vector, make it one.
        b = b';
    end
    
    % Get size of vector b.
    [mb, ~] = size(b);
end

% Check that s is a vector and get its size.
if ~isvector(s)
    error('Argument s is not a vector!');
else
    if isrow(s)     % If not a column vector, make it one.
        s = s';
    end
    
    % Get size of vector s.
    [ms, ~] = size(s);
end

% Check that sizes of matrix and vector inputs are correct.
if nD ~= mb
    error('Number of columns in D do not match length of vector b!');
elseif mD ~= ms
    error('Number of rows in D does not match length of vector s!');
else
    % Set final dimensions.
    m = mD;
    n = nD;
end

% Check that the given options is actually a struct...
if ~isstruct(options)
    error('Given options is not a struct! At least pass empty struct!');
end

end
% -------------------------------------------------------------------------