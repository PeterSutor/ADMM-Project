% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Solves the Quadratic Program problem. Quadratic Programming minimizes for
% x the objective function:
%    obj(x) = 1/2*<x,P*x> + <q,x> + r = 1/2*x^T*P*x + q^T*x + r, 
%    subject to D*x = s, x >= 0
% where D is a matrix and s is a column vector of appropriate length. Thus,
% x and b are column vectors of the same length. Matrix P and vector q 
% represent coefficients in the strictly quadratic and strictly linear 
% parts of the program. The value r is a constant. We assume P is a square,
% nonnegative matrix. Note that this formulation is known as the standard 
% form for a quadratic program. One can use any conic constraint on x, not 
% just x >= 0. In this case, one will have to provide the appropriate 
% proximal function for g to minimize this in the options struct 
% (options.altproxg).
%
% Alternatively, we can simplify the constraint in our objective function 
% to be:
%    obj(x) = 1/2*<x,P*x> + <q,x> + r = 1/2*x^T*P*x + q^T*x + r, 
%    subject to lb <= x <= ub
% In this case of the Quadratic Programming problem, lb is a lower bounding
% vector and ub an upper bounding vector. We refer to this constraint
% formulation as 'bounded'. The former one is referred to as 'standard'.
% 
% The ADMM Augmented Lagrangian here is:
%    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + 
%                   constant(u),
% where f(x) = 1/2*x^T*P*x + q^T*x + r such that x is in the set 
% {x: D*x = s} and g(z) is the indicator function that z is not in the 
% non-negative orthant (R_+^n). The rho in L_rho is the dual step size 
% parameter. ADMM solves the equivalent problem of minimizing: 
%    f(x) + g(z), subject to x - z = 0.
%
% ADMM requires proximal operators for minimizing L_rho(x,z,u) for x and z,
% separately - i.e., proximal operators for functions f and g. The proximal
% operators can be obtained in exactly the same method as in a Linear 
% Program (see linearprogram.m), except with the inclusion of the matrix P
% in the solution. That is, the proximal operator for f is:
%    [ P + rho*I, D^T ]   [ x ]   [ q - rho*(z - u) ]    [0]
%    [     D    ,  0  ] * [ y ] + [       -s        ] =  [0]
% The solution being (now that the problem is square and has an inverse:
%        [ x ]          ([ P + rho*I, D^T ])   [ rho*(z - u) - q ]
%    v = [ y ] = inverse([     D    ,  0  ]) * [        s        ]
% Our minimized x in the proximal function is now the x-part in solution 
% vector v.
% 
% For the proximal function for g, our proximal operator is simple. To 
% project a given vector v = x + u into the non-negative orthant, we simply
% take the positive parts of v and set the rest to 0; this is our minimized
% vector z.
%
% Alternatively, we can simplify the constraint in our objective function 
% to be:
%    obj(x) = 1/2*<x,P*x> + <q,x> + r = 1/2*x^T*P*x + q^T*x + r, 
%    subject to lb <= x <= ub
% In this case, we can take a different approach. We assume that P can be 
% Cholesky factored into P = X^T*X. Furthermore, if we perturb P with 
% rho*I, then there must exist a square matrix R such that P + rho*I = 
% R^T*R, via the same Cholesky factorization. Then, we can minimize for 
% function f by taking the gradient and solving for x, when set to 0:
%    grad_x(L_rho(x,z,u)) := 0 
%       <--> grad_x(1/2*x^T*(X^T*X)*x + q^T*x + r) + rho*(x - z + u) = 0
%       <--> grad_x(1/2*||X*x||^2) + q + rho*(x - z + u) = 0
%       <--> X^T*X*x + rho*x + [q - rho*(z - u)] = 0
%       <--> (X^T*X + rho*I)*x = rho*(z - u) - q
%       <--> (R^T*R)*x = rho*(z - u) - q                                (1)
%       <--> R^T*(R*x) = rho*(z - u) - q
%       <--> R^T*y = rho*(z - u) - q                                    (2)
% Assuming we know R, (1) can be efficiently solved by solving (2) for y, 
% and then solving R*x = y for x. For the proximal operator for g, where g
% in this situation indicates whether z is NOT in the set 
% {z: lb <= z <= ub}, thus the minimizing z for function g is simply a 
% min-max between z and the bounds lb and ub:
%    z_i := min(ub_i, max(lb_i, v_i))
% where v = x + u.
% 
% This solver will automatically determine which constraint type you want
% to use based on your inputs into cons1 and cons2. If cons1 is a matrix,
% it is assumed that D = cons1 and s = cons2. Likewise, if cons2 is a
% matrix, it is assumed D = cons2 and s = cons2. Finally, if both cons1 and
% cons2 are vectors, it is assumed they are the lower bound lb and upper
% bound ub. It doesn't matter in which order they are provided, they will
% be checked to be sure that either cons1 <= cons2 or the reverse.
%
% Check the quadraticprogram section of function getproxops to see the 
% proximal operators.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the quadraticprogram function is executed with no inputs, it 
% will run quadraticprogramtest with no inputs. This will generate a random
% problem of size m = n = 2^7, and test it for correctness, showing the
% results.

function results = quadraticprogram(P, q, r, cons1, cons2, options)
% INPUTS ------------------------------------------------------------------
% P:        The square coefficient matrix of size n by n in the objective
%           function above.
% q:        The coefficient matrix of length n in the objective function
%           above.
% r:        The constant r in the objective function above.
% cons1:    The first constraint parameter. If this is a matrix, it is
%           assumed that you wish to solve the standard constraint problem.
%           Thus, if cons2 is not a vector, an error is returned as output.
%           If it is a vector, then it is either s (based on if cons2 is a
%           matrix), or either lb or up (based on if cons2 is a vector and
%           is strictly greater or equal to cons1, or strictly less than or
%           equal to, respectively).
% cons2:    The second constraint parameter. If this is a matrix, it is
%           assumed that you wish to solve the standard constraint problem.
%           Thus, if cons1 is not a vector, an error is returned as output.
%           If it is a vector, then it is either s (based on if cons2 is a
%           matrix), or either lb or up (based on if cons2 is a vector and
%           is strictly greater or equal to cons1, or strictly less than or
%           equal to, respectively).
% options:  A struct containing options customizing the ADMM execution. If
%           no options are provided (an empty or irrelevant struct is given
%           as the options parameter), default settings are applied, as 
%           mentioned in the user manual. If you'd like an alternate conic
%           constraint parameter, set options.altproxg to an alternate
%           proximal operator for g with inputs x, z, u and rho, which
%           computes the minimizing z for your conic constraint under
%           vector v = x + u.
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
% quadraticprogramtest function with no arguments. Returns test results for 
% random data of size m = 2^6 and n = 2^7.
if nargin == 0
    display(['No arguments detected; running a demo test of the', ...
        ' Quadratic Progam problem for random square data of m = 2^6', ...
        ' and n = 2^7:']);
    results = quadraticprogramtest();
    results.solverruntime = toc;                % End timing.
    
    return;
end

% Run a check for errors on inputs, returning potentially fixed outputs.
[m, n, P, q, cons1, cons2, constraint] = ...
    errorchecker(P, q, r, cons1, cons2, options);

% Determine if user specified a rho value. If not, set to default.
if ~isfield(options, 'rho')
    startrho = 1.0;
else
    startrho = options.rho;
end

args.rho = startrho;

% Set up solver data for either standard or bounded constraint forms, to
% pass into function getproxops, which returns the appropriate proximal
% operators.
if strcmp(constraint, 'standard')
    % Set up data for getproxops to return proximal functions.
    Dt = cons1';            % Cache transpose of data D.
    In = eye(n);            % Cache properly sized identity matrix.
    zero = zeros(m);        % Cache properly sized zero matrix.

    % Populate the data for the args struct to pass to getproxops.
    args.D = cons1;
    args.Dt = Dt;
    args.In = In;
    args.zero = zero;
    args.P = P;
    args.q = q;
    args.s = cons2;
    args.n = n;
    args.constraint = constraint;
    args.rho = startrho;
elseif strcmp(constraint, 'bounded')
    % Populate the data for the args struct to pass to getproxops.
    args.P = P;
    args.q = q;
    args.lb = cons1;
    args.ub = cons2;
    args.rho = startrho;
    args.n = n;
    args.constraint = constraint;
end

% If the user provided an alternate proximal function for g (perhaps they
% have a different conic constraint than x >= 0 for the problem), use that
% proximal function. Otherwise, use the default.
if (isfield(options, 'altproxg') && ...
    isa(options.altproxg, 'function_handle'))

    % Set proximal operators.
    [minx, ~] = getproxops('quadraticprogram', args);
    minz = options.altproxg;
else
    % Get proximal operators.
    [minx, minz] = getproxops('quadraticprogram', args);
end

% Set constraints and objective function for ADMM.
options.A = 1;
options.B = -1;
options.c = 0;
options.m = n;
options.nA = n;
options.nB = n;
options.obj = @(x, z) 1/2*x'*P*x + q'*x + r;   % The objective function.

% Perform ADMM on this setup.
results = admm(minx, minz, options);
results.solverruntime = toc;                    % End timing.

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Simply checks for any user error on same input, stops if error is
% detected. If input is correct, return info needed for problem.

function [m, n, P, q, cons1, cons2, constraint] = ...
    errorchecker(P, q, r, cons1, cons2, options)
% INPUTS ------------------------------------------------------------------
% Same as for the quadraticprogram function.
% 
% OUTPUTS -----------------------------------------------------------------
% m:    The number of rows in matrix D in standard form. Set to n in
%       bounded form as it is not a used value in that constraint form.
% n:    The size of the n by n square matrix P and column vector of length
%       n, q, if using bounded form. Otherwise, it is the number of
%       columns in matrix D in standard form.
% The rest are the same as the inputs for this function, but corrected for
% any weird input.
% -------------------------------------------------------------------------


% Perform error checks on input...
P = errorcheck(P, 'issquare', 'P');
q = errorcheck(q, 'iscolumnvector', 'q');
errorcheck(r, 'isnumber', 'r');
errorcheck(options, 'isstruct', 'options');

% Obtain sizes...
[nP, ~] = size(P);
mq = length(q);

% Check that sizes match...
if (mq ~= nP)
    error('The dimensions of square matrix P and vector q do not match!');
end

% Check if the datatypes of constraints are correct. Then determine whether
% the input is in standard or bounded constraint form. For bounded,
% normalize the input so that the lower bound is cons1 and upper bound
% cons2. For standard, normalize so that cons1 is matrix D and cons2 is
% vector s.
if ~isnumeric(cons1) || ~ismatrix(cons1) || ...
    ~isnumeric(cons2) || ~ismatrix(cons2)
    error('Constraint inputs are not numeric vectors or matrices!');
% Bounded constraint case (two vectors as input, an lb and ub)...
elseif (size(cons1, 1) == 1 || size(cons1, 2) == 1) && ...
    (size(cons2, 1) == 1 || size(cons2, 2) == 1)
    constraint = 'bounded';             % Remember the constraint type.
    n = nP;                             % In this case, n is size of P.
    m = n;
    
    % Use error check to get both constraints into column vector form.
    cons1 = errorcheck(cons1, 'iscolumnvector', 'cons1');
    cons2 = errorcheck(cons2, 'iscolumnvector', 'cons2');
    
    % Check that the lengths match between constraints. Check whether cons2
    % is actually lower bound and swap cons1 with cons2 if so. Check
    % whether the constraints are actually lower and upper bounds; if not
    % return an error.
    if length(cons1) ~= length(cons2)
        error(['Lengths of lower and upper bound constraints on', ...
            ' solution x do not match!']);
    elseif length(cons1) ~= nP
        error(['Bound vectors do not match predicted', ...
            ' length of solution x!']);
    elseif isequal(max(cons1, cons2), cons1)
        temp = cons1;
        cons1 = cons2;
        cons2 = temp;
    elseif ~isequal(max(cons1, cons2), cons2)
        error(['Given constraint variables do not specify an upper', ...
            ' and lower bound on solution x!']);
    end
% Standard constraint case (one matrix D and one vector s)...
elseif (size(cons1, 1) == 1 || size(cons1, 2) == 1) || ...
    (size(cons2, 1) == 1 || size(cons2, 2) == 1)
    constraint = 'standard';            % Remember the contraint type.
    
    % Swap cons1 and cons2 if it appears that cons1 is s.
    if (size(cons1, 1) == 1 || size(cons1, 2) == 1)
        temp = cons1;
        cons1 = cons2;
        cons2 = temp;
    end
    
    % Force cons2 into a column vector form if necessary and obtain lengths
    % of D = cons1 and s = cons2.
    cons2 = errorcheck(cons2, 'iscolumnvector', 'cons2');
    ms = length(cons2);
    [mD, nD] = size(cons1);
    
    % Make sure the lengths of D = cons1 and s = cons2 match up in problem
    % definition and with P and q.
    if nD ~= mq
        error(['Number of columns in constraint matrix in standard', ...
            ' form do not match lengths of P and q!']);
    elseif mD ~= ms
        error(['Number of rows in constraint matrix in standard form', ...
            ' does not match length of constraint', ...
            ' vector!\n(D and s in standard form)'], 1);
    else
        n = nD;                             % Assign n as columns in D.
        m = mD;                             % Assign m as rows in D.
    end
% Otherwise, something is wrong with constraint inputs. Likely that they
% are both matrices...
else
    error(['It appears that both constraint inputs are matrices!\nIf', ...
        ' trying to use standard constraint form,', ...
        ' only one can be a matrix.'], 1);
end

end
% -------------------------------------------------------------------------