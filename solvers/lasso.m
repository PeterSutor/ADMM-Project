% Coded by:     Peter Sutor Jr.
% Last edit:    5/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Solves the LASSO problem using ADMM. LASSO minimizes for x the objective
% function:
%    obj(x) = 1/2*||D*x - s||_2^2 + lambda*||x||_1, 
% where D is a data matrix and s is a signal column vector of appropriate
% length. Thus, x is a column vector of the same length. The parameter
% lambda here is the regularization parameter.
% 
% The ADMM Augmented Lagrangian here is:
%    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + constant(u),
% where f(x) is the first term in the objective, and g(z) the second, with
% z = x. The rho in L_rho is the dual step size parameter. ADMM solves the
% equivalent problem of minimizing: f(x) + g(z), subject to x - z = 0.
%
% To solve this problem, we need to supply ADMM with proximal operators for
% f and g. We can determine these as follows:
%
% For f, we proceed by standard minimization and take the gradient of the
% Augmented Lagrangian, setting it equal to 0 and solving for the
% minimizing x:
%    grad_x(L_rho(x,z,u)) = grad_x(f(x)) + rho*(x - z + u)
%                         = D^T*(D*x - s) + rho*(x - z + u) := 0
% This implies:
%    D^T*D*x + rho*x = (D^T*D + rho*I)*x = D^T*s + rho*(z - u)
% Thus, the minimizing x is:
%    x := (D^T*D + rho*I)^{-1}[D^T*s + rho*(z - u)]                     (1)
% To efficiently compute this, we precompute and save D^T*D, D^T*s and the
% Cholesky decomposition L*U = R*R' = D^T*D + rho*I. The addition of rho*I
% can be efficiently computed by vectorizing the addition of rho along the
% diagonal of the cached D^T*D. The linear system in (1), can be expressed
% as:
%    (D^T*D + rho*I)*x = D^T*s + rho*(z - u)
% By substitution:
%    L*U*x = D^T*s + rho*(z - u)                                        (2)
% Let U*x = y. Then, we see that solving (2) for x is equivalent to solving
% the linear system L*y = D^T*s + rho*(z - u) for y, which can be done
% efficiently using Matlab's backslash operator, and then solving the
% system U*x = y for x, by the same means. Note that there are no dimension
% restrictions on D; D^T*D will always be square. If D is a very tall
% matrix, D^T*D will be a very small square matrix. If D is very fat,
% D*D^T will be a very small square matrix. To take advantage of this fact,
% we can use D*D^T by utilizing the Matrix Inversion theorem, and solving
% the system by:
%    x := (I - D^T*(D*D^T)^{-1}*D)*(z - u) + D^T*(D*D^T)^{-1}*s
% Inverting D*D^T can be done efficiently in the same manner as above.
% 
% For g, we simply minimize this using soft-thresholding on v = u + z and
% t = lambda/rho.
%
% As an alternative, we can solve this problem in a parallel manner, by
% performing consensus LASSO. In consensus LASSO, we split up our problem
% matrix D and signal vector s into row segments (called slices), that can
% be of varying size. Then, we simply perform the above algorithm (normal
% LASSO) on the slice of data. If we do this over all the slices (in
% parallel), then we can average the solutions on each slice to give the
% consensus solution. In general, this is called Group LASSO. Thus, the
% x-minimization step is done in parallel, with each solution performing
% LASSO on their data subset. For each slice i, the x_i that is computed,
% along with the prior u_i is summed and averaged along all slices, to give
% the z-update. Each u_i is subsequently updated using this z and
% individual x_i and prior u_i.
%
% Check the LASSO section of function getproxops to see the code for the
% proximal operators.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the lasso function is executed with no inputs, it will run
% lassotest with no inputs. This will generate a random problem of size 
% m = 2^8, n = 2^6, and tests it for correctness, showing the results.

function [results] = lasso(D, s, lambda, options)
% INPUTS ------------------------------------------------------------------
% D             The m by n matrix in the Lasso problem.
% s             n = length(s) vector of data in the Lasso problem.
% lambda        Regularization parameter Lasso problem.
% options       Options argument for ADMM.
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

% If no arguments are given, run a demo test by running the lassotest
% function with no arguments. Creates a test as described in the NOTE above
% in the description.
if nargin == 0
    display(['No arguments detected; running a demo test of the', ...
        ' LASSO problem for random data of size m = n = 2^8:']);
    results = lassotest();
    results.solverruntime = toc;                % End timing.
    
    return;
end

% Error checking and fixing input.
lambda = errorcheck(lambda, 'isnonnegativereal', 'lambda');
D = errorcheck(D, 'ismatrix', 'D');
s = errorcheck(s, 'isvector', 's');

% Obtain and check rho.
if isfield(options, 'rho')
    rho = errorcheck(options.rho, 'ispositivereal', 'options.rho');
else
    rho = 1.0;
end

% Decide whether to use parallel ADMM from user input.
if isfield(options, 'parallel')
    if strcmp(options.parallel, 'both') || ...
        strcmp(options.parallel, 'zming') || ...
        strcmp(options.parallel, 'xminf')
        options.parallel = 'none';
        options.stopcond = 'both';
        parallel = 1;
    else
        parallel = 0;
    end
else
    parallel = 0;
end

% Perform the serial or parallel algorithm as specified.
if ~parallel
    Dts = D'*s;                 % Stores product of transpose of A times b.
    [m, n] = size(D);           % Gets the dimensions of our problem.
    Id_n = speye(n);            % Stores the sparse n by n identity matrix.
    Id_m = speye(m);            % Stores the sparse m by m identity matrix.
    
    % Get an LU decomposition for the x-minimization step.
    if(m >= n)                  % A is square or tall.
        % Get lower triangular L for traspose of A times A plus rho for n.
        L = chol(D'*D + rho*Id_n, 'lower');
    else                        % A is short and fat.
        % Instead, scale by rho and swap roles of A and A transpose. Need
        % to use m by m identity matrix in this case.
        L = chol(1/rho*(D*D') + Id_m, 'lower');
    end

    L = sparse(L);              % Sparse version of our L.
    U = sparse(L');             % Upper triangular U is trivial; simply the
                                % transpose of L. Made sparse.

    % Populate the args struct for call to getproxops, which returns our
    % proximal operators.
    args.D = D;
    args.Dts = Dts;                             
    args.L = L;
    args.U = U;
    args.m = m;
    args.n = n;
    args.lambda = lambda;
    args.parallel = parallel;
    args.rho = rho;
    
    % Get proximal operators for the LASSO problem.
    [minx, minz] = getproxops('LASSO', args);
else
    % Obtain user given slices or set to 0 to indicate distributed workload
    % among workers.
    if isfield(options, 'slices')
        slices = options.slices(1);
    else
        slices = 0;
    end
    
    % Set up parallel pool...
    pool = gcp;
    
    % Process slices argument.
    sliceopts.slicelength = size(D, 1);
    sliceopts.workers = pool.NumWorkers;
    slices = errorcheck(slices, 'slices', 'options.slices', sliceopts);
    
    % Set args parameters.
    args.slices = slices;
    args.D = D;
    args.s = s;
    args.lambda = lambda;
    args.rho = rho;
    args.parallel = parallel;
    
    [~, n] = size(D);
    
    % Get proximal operators for the LASSO problem.
    [minx, minz, extra] = getproxops('LASSO', args);
    options.altu = extra.altu;
    options.specialnorms = extra.specialnorms;
end

% Our objective function.
obj = @(x, z) 0.5*sum((D*x - s).^2) + lambda*norm(z, 1);

% Set up the options struct. Constants for constraint parameters, for
% efficiency.
options.obj = obj;
options.A = 1;
options.At = 1;
options.m = n;
options.nA = n;
options.nB = n;
options.B = -1;
options.c = 0;
options.parallel = 'none';

% Solve the LASSO problem with our ADMM function.
[results] = admm(minx, minz, options);
results.solverruntime = toc;

end
% -------------------------------------------------------------------------