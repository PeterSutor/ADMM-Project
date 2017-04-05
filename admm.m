% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Performs the Alternating Direction Method of Multipliers (ADMM) algorithm
% on given input. ADMM minimizes f(x) + g(z) such that Ax + Bz = c, where
% the matrix A is of size m by n, matrix B is of size m by n, and c is a
% row vector of size m. The functions f and g must be convex. This 
% particular ADMM uses scaled dual variables.
%
% Requires given argument minimizing functions of the augmented Lagrangian
% function: 
%    L_rho(x,z,u) = L_rho(x,z,y) = 
%       f(x) + g(z) + y^T(Ax + Bz - c) + (rho/2)(||Ax + Bz - c||_2)^2 = 
%       f(x) + g(z) + (rho/2)(||Ax + Bz - c + u||_2)^2 + constant(u), 
% where u = y/rho (the scaled dual variable). One function returns the 
% minimizing x (function xminf), the other z (zming). The constant(u) in 
% L_rho(x,z,u) is unnecesary as it does not affect the minimization points.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.

function results = admm(xminf, zming, options)
% INPUTS ------------------------------------------------------------------
% xminf:    A function handle of form xminf(x, z, u) that returns the 
%           optimal x minimizing L_rho(x,z,u) (i.e., the proximal operator 
%           for f).
% zming:    A function handle of form zming(x, z, u) that returns the
%           optimal z minimizing L_rho(x,z,u) (i.e., the proximal operator 
%           for g).
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

% Check that the given options is actually a struct...
if ~isstruct(options)
    error('Given options is not a struct! At least pass empty struct!');
end

adaptive = setopt(options, 'adaptive', 0);
%ddf = setopt(options, 'ddf', options.ddf, 0);

% Set parameters according to the options struct, or default value if
% unspecified.
quiet = setopt(options, 'quiet', 1);
rho = setopt(options, 'rho', 1.0);
N = setopt(options, 'maxiters', 1000);
domaxiters = setopt(options, 'domaxiters', 0);
relax = setopt(options, 'relax', 1);
parallel = setopt(options, 'parallel', 'none');
slices = setopt(options, 'slices', 0);
fast = setopt(options, 'fast', 0);
fasttype = setopt(options, 'fasttype', 'weak');
obj = setopt(options, 'obj', 0);
objevals = setopt(options, 'objevals', 0);
convtest = setopt(options, 'convtest', 0);
convtol = setopt(options, 'convtol', 1e-10);
stopcond = setopt(options, 'stopcond', 'standard');
nodualerror = setopt(options, 'nodualerror', 0);
ABSTOL = setopt(options, 'abstol', 1e-5);
RELTOL = setopt(options, 'reltol', 1e-3);
HNORMTOL = setopt(options, 'Hnormtol', 1e-6);
m = setopt(options, 'm', 0);
nA = setopt(options, 'nA', 0);
nB = setopt(options, 'nB', 0);

% Set vector c in constraint A*x + B*z = c based on options struct.
if isfield(options, 'c')                    % Case user specified it.
    c = options.c;                          % Set c from options.
    
    if (isvector(c) || c == 0)              % If c is a vector or constant.
        if (isrow(c))                       % If c is a row vector...
            c = c.';                        % Make it a column vector.
        end
        
        % Reported error if we can't figure out the size of vector c.
        if (m == 0 && length(c) == 1)
            error(['Given vector c is scalar and no length m has been', ...
                ' provided; unable to infer m - please specify it in ', ...
                'options struct.']);
        elseif length(c) ~= 1               % Otherwise...
            m = length(c);                  % Set m to length of given c.
        end
    else
        % Error because c isn't even a vector!
        error('Given c in constraint Ax + Bz = c is not a vector!');
    end
else                                        % Case c unspecified by user.
    % Ensure that m is a positive integer, otherwise report likely error.
    if (m > 0)
        if (floor(m) == m)
            c = zeros(m, 1);                % Set to m-long zero vector.
        else
            error('Noninteger size m of c in constraint Ax + Bz = c!');
        end
    else
        error('Must specify a vector c in constraint Ax + Bz = c!');
    end
end

% Set matrix A in constraint A*x + B*z = c based on options struct.
if isfield(options, 'A')                    % Case user specified A.
    A = options.A;
    
    % Check if A is a matrix or function handle and assign appropriately.
    if ismatrix(A)                          % Case that a matrix was given.
        [mA, nAtemp] = size(A);             % Get dimensions of given A.
        options.At = A';                    % Cache the transpose.
        A = @(v) A*v;                       % Convert to function handle.
    elseif isa(A, 'function_handle')        % Case A is a function handle.
        % If user didn't specify dimensions of A, we can't proceed.
        if nA == 0
            error(['Matrix A is a function handle, but no number of ', ...
                'columns nA specified for it; cannot infer nA - ', ...
                'please specify it in options struct!']);
        end
        
        % Get dimensions of given A.
        [mA, nAtemp] = size(A(zeros(nA, 1)));
    else                                    % Otherwise, report an error.
        error(['Given A in constraint Ax + Bz = c is neither a ', ... 
            'numeric matrix nor function handle of single vector!']);
    end
    
    % If dimensions don't match up for non-scalar A, we can't proceed and
    % we report and error.
    if (mA ~= m && mA ~= 1)
        error(['Number of rows in matrix A do not match length of ', ...
            'column vector c in constraint Ax + Bz = c']);
    end
    
    % If A is a scalar (for efficiency) but we don't know the intended
    % dimensions of it, we can't proceed and report an error.
    if (nA == 0 && nAtemp == 1 && mA == 1)
        error(['Given scalar as matrix A with no number of columns nA', ... 
            'specified in options struct; cannot infer nA - please ', ...
            'specify nA in options!']);
    % Go with nA, if A's dimensions not specified explicitly but implicitly
    % by vector c in constraint A*x + B*z = c, as the number of columns.
    elseif (nAtemp ~= 1 && nA ~= nAtemp)
        nA = nAtemp;
    end
else
    % We weren't given a matrix A for our contraint and we can't proceed,
    % reporting the error.
    error('Must specify a matrix A in constraint Ax + Bz = c!');
end

% Assign the transpose of A based on the options specified by user.
if isfield(options, 'At')                   % If specified in options...
    At = options.At;                        % Assign the transpose.
    
    % Check if given transpose is valid.
    if ismatrix(At)                         % Case a matrix is given.
        [nAt, mAt] = size(At);              % Get dimension of transpose.
        At = @(v) At*v;                     % Convert to function handle.
    elseif isa(At,'function_handle')        % Case given a function handle.
        % Get the dimensions of A^T by checking size of all zero input.
        [nAt, mAt] = size(At(zeros(mA, 1)));
    else
        % We weren't given a usable A^T and can't proceed, reporting an
        % error.
        error(['Given At (A transpose) in constraint Ax + Bz = c is ', ... 
            'neither a numeric matrix nor function ', ...
            'handle of single vector!']);
    end
    
    % If the number of columns do not match up, we can't proceed and report
    % an error.
    if (mAt ~= mA)
        error(['Number of columns in At (A transpose) does not match ', ... 
            'number of rows in A, in constraint Ax + Bz = c']);
    end
    
    % If the number of rows do not match up for non-scalar matrix
    % transpose, we can't proceed and report an error.
    if (nAt ~= nA && ~(nAt == 1 && mAt == 1))
        error(['Number of rows in At (A transpose) do not match ', ... 
            'number of columns in A, in constraint Ax + Bz = c']);
    end
else
    % No matrix A was specified and transpose could not be computed.
    error('Must specify a matrix A in constraint Ax + Bz = c!');
end

% Set matrix B in constraint A*x + B*z = c based on options struct.
if isfield(options, 'B')                    % Case specified by options.
    B = options.B;                          % Assign B appropriately.
    
    % Check if given matrix B is valid.
    if ismatrix(B)                          % Case B is a matrix.
        [mB, nBtemp] = size(B);             % Get dimensions of B.
        B = @(v) B*v;                       % Convert to function handle.
    elseif isa(B,'function_handle')         % Case given a function handle.
        if nB == 0
            % We don't know the dimensions of B from a function handle,
            % thus we can't proceed and report an error.
            error(['Matrix B is a function handle, but no number of ', ...
                'columns nB specified for it; cannot infer nB - ', ...
                'please specify it in options struct!']);
        end
        
        % Get the dimensions of B from zero vector as input.
        [mB, nBtemp] = size(B(zeros(nB, 1)));
    else
        % We weren't given a usable matrix B, thus we can't proceed and
        % report an error.
        error(['Given B in constraint Ax + Bz = c is neither a ', ... 
            'numeric matrix nor function handle of single vector!']);
    end
    
    % If row dimensions do not match up for non-scalar B (for efficiency),
    % we can't proceed and report an error.
    if (mB ~= m && mB ~= 1)
        error(['Number of rows in matrix B do not match length of ', ...
            'column vector c in constraint Ax + Bz = c']);
    end
    
    % If we weren't given the column dimensions, and a scalar B (for 
    % efficiency), we can't proceed and report an error.
    if (nB == 0 && nBtemp == 1 && mB == 1)
        error(['Given scalar as matrix B with no number of columns nB', ... 
            'specified in options struct; cannot infer nB - please ', ...
            'specify nB in options!']);
    % Go with nB, if B's dimensions given by user don't make sense, as
    % those have been checked.
    elseif (nBtemp ~= 1 && nB ~= nBtemp)
        nB = nBtemp;
    end
else
    % We weren't given a valid matrix B and thus we can't proceed,
    % reporting an error.
    error('Must specify a matrix B in constraint Ax + Bz = c!');
end

% Double check that we can evaluate the objective function given to us.
canEvalObj = (objevals && isa(obj, 'function_handle'));

% Set inital values for iterations, if specified, defaulting to zero
% vectors otherwise.
x = setopt(options, 'x0', zeros(nA, 1));
z = setopt(options, 'z0', zeros(nB, 1));
u = setopt(options, 'u0', zeros(m, 1));

% Set initial values in reported results.
results.x0 = x;
results.z0 = z;
results.u0 = u;

% Default algorithm is 0 - standard ADMM for now.
alg = 0;

% Check if the user wanted to use Fast ADMM techniques. Check Tom
% Goldstein's paper on Fast / Accelerated ADMM to read about these
% techniques in detail.
if fast
    % Assign basic variables for Fast / Accelerated ADMM.
    v = z;                              % Accelerated z variable.
    uhat = u;                           % Accelerated u variable.
    acurr = 1;                          % Current value of alpha.
    aprev = 1;                          % Previous value of alpha.
    
    % If the user specified weak duality, we perform Accelerated ADMM. This
    % is the default fast behavior.
    if (strcmp(fasttype, 'weak'))
        % Initialize d.
        d = Inf;                        % Some current d value.
        dprev = Inf;                    % Some previous d value.
        
        % Set the probability n of restarting in Accelerated ADMM.
        n = setopt(options, 'restart', 0.999);
        
        % If user gives a weird restart probability, set to default.
        if (n <= 0 || n >= 1)
            n = 0.999;
        end
        
        % The user specified relative tolerance for d values.
        DVALTOL = setopt(options, 'dvaltol', 1e-8);
        
        results.dvaltol = DVALTOL;      % Report tolerance in results.
        
        alg = 2;                        % Set algorithm type to AADMM.
    else
        alg = 1;                        % Set algorithm type to FADMM. 
    end
end

% Checks if the user allows convergence testing and is fine with using both
% types of stop conditions.
if (convtest || strcmp(stopcond, 'hnorm') || strcmp(stopcond, 'both'))
    % Defines the H-norm squared function that is used to check convergence
    % of w residuals.
    H_norm_sq = @(wdiff) rho*norm(B(wdiff(nA+1:nA + nB, :)), 'fro')^2 + ...
        rho*norm(wdiff(nA + nB + 1:nA + nB + m, :), 'fro')^2;
    
    % A vector representing the iteration data for ADMM for some iteration.
    w = [x; z; rho*u];
    
    % Absolute difference tolerance of H norm values squared.
    results.Hnormtol = HNORMTOL;
end

start = tic;                                % Start timing execution.

% If we can report results to the screen...
if ~quiet
    if canEvalObj
        % Print headers for iter's contents.
        fprintf('%7s\t%20s\t%20s\t%20s\t%20s\t%20s\n', 'Iteration', ...
          'Primal Residual Norm', 'Primal Error', 'Dual Residual Norm', ...
          'Dual Error', 'Objective Value');
    else
        % Print headers for iter's contents.
        fprintf('%7s\t%20s\t%20s\t%20s\t%20s\n', 'Iteration', ...
          'Primal Residual Norm', 'Primal Error', 'Dual Residual Norm', ...
          'Dual Error');
    end
end

% Check if a valid maximum number of iterations was given, otherwise
% setting it to default.
if N > 0
    % Get the real component of ceiling, in case any weird input is given.
    N = real(ceil(N));
else
    N = 1000;                               % Default value.
end


% In the case of parallel ADMM...
if strcmp(parallel, 'xminf') || strcmp(parallel, 'zming') || ...
    strcmp(parallel, 'both')

    % Create the worker pool and find out how many workers are available.
    pool = gcp('nocreate');
    workers = pool.NumWorkers;
    
    % Report error if there are not enough workers to perform a parallel
    % computation. Otherwise, initialize the parallel pool.
    if workers <= 0
        error(['There are no workers on this machine,', ...
            ' cannot perform parallel ADMM!']);
    else
        gcp;
    end
    
    % Error check on settings for slices and parallelization.
    if ~iscell(slices) && strcmp(parallel, 'both')
        error(['For parallelizing both proximal ops, please:\n', ...
            '\tSpecify slices for f as a vector in options.slices.\n', ...
            '\tSpecify slices for g as a vector in options.slices.'], 1);
    elseif iscell(slices) && ~strcmp(parallel, 'both')
        error(['Trying to parallelize both proximal operators,', ...
            ' but options.slices is not a 2 element cell!']);
    elseif ~isvector(slices) && (strcmp(parallel, 'xminf') || ...
        strcmp(parallel, 'zming'))
        error(['Option options.slices should be a vector for', ...
            ' parallelizing one proximal operator!']);
    end
    
    % Set up slices for parallel proximal functions for minimizing x and z.
    if iscell(slices)
        slicesx = slices{1};
        slicesz = slices{2};
    elseif isvector(slices) && strcmp(parallel, 'xminf')
        slicesx = slices;
        slicesz = [];
    elseif isvector(slices) && strcmp(parallel, 'zming');
        slicesx = [];
        slicesz = slices;
    end
    
    if strcmp(parallel, 'xminf') || strcmp(parallel, 'both')
        % Make xminf a parallelized proximal function for f, and the given
        % xminf the function for decomposition i of proximal update.
        xminfi = xminf;
        xminf = @parproxf;
        
        % Validate slices.
        optsx.workers = workers;
        optsx.slicelength = length(x);
        slicesx = errorcheck(slicesx, 'slices', 'x-slices', optsx);
    end
    
    if strcmp(parallel, 'zming') || strcmp(parallel, 'both')
        % Make zming a parallelized proximal function for g, and the given
        % zming the function for decomposition i of proximal update.
        zmingi = zming;
        zming = @parproxg;
        
        % Validate slices.
        optsz.workers = workers;
        optsz.slicelength = length(z);
        slicesz = errorcheck(slicesz, 'slices', 'z-slices', optsz);
    end
end

    % ---------------------------------------------------------------------
    % DESCRIPTION ---------------------------------------------------------
    %
    % A function that parallelizes the x-update over the proximal operator,
    % over the slices provided / specified by user, using parfor.
    
    function xmin = parproxf(x, z, u, rho)
    % Same inputs and outputs as proximal operator for f.
        
        % This cell array stores the component results per each slice.
        mins = cell(length(slicesx), 1);
        
        % Purpose: Obtain the slices of the minimized x from the decomposed
        % proximal operator for f, xminfi. Since parfor is parallel, this
        % obtains speedups proportional to number of workers. As parallel
        % processing is local, it is fine to pass vectors x, z, and u to
        % each worker, as they are passed by reference and shouldn't be
        % modified in the function call to xminfi. The function xminfi
        % should return the corresponding splice of the proximal operator
        % specified by k.
        parfor k = 1:length(slicesx)
            mins{k} = feval(xminfi, x, z, u, rho, k);
        end
        
        xmin = cell2mat(mins);      % Collect results into single vector.
        
    end
    % ---------------------------------------------------------------------
    
    
    
    % ---------------------------------------------------------------------
    % DESCRIPTION ---------------------------------------------------------
    %
    % A function that parallelizes the z-update over the proximal operator,
    % over the slices provided / specified by user, using parfor.
    
    function zmin = parproxg(x, z, u, rho)
    % Same inputs and outputs as parallel proximal operator for f.
        
        % This cell array stores the component results per each slice.
        mins = cell(length(slicesz), 1);
        
        % Purpose: Obtain the slices of the minimized z from the decomposed
        % proximal operator for g, zmingi. Since parfor is parallel, this
        % obtains speedups proportional to number of workers. As parallel
        % processing is local, it is fine to pass vectors x, z, and u to
        % each worker, as they are passed by reference and shouldn't be
        % modified in the function call to zmingi. The function zmingi
        % should return the corresponding splice of the proximal operator
        % specified by k.
        parfor k = 1:length(slicesz)
            mins{k} = feval(zmingi, x, z, u, rho, k);
        end
        
        zmin = cell2mat(mins);      % Collect results into single vector.
        
    end
    % ---------------------------------------------------------------------
 
% If the user requires some preprocessing before ADMM iteration runs, they
% can pass a function handle for a preprocess function. This function, when
% called, performs some local preprocessing in the function calling ADMM.
if (isfield(options, 'preprocess') && ...
    isa(options.preprocess, 'function_handle'))
    options.preprocess();
end

dim = size(x);              % Determine size of x.

% If size is a default vector, dimension is 1, else just the length.
if (length(dim) == 2 && dim(2) == 1)
    dim = 1;
else
    dim = length(dim);
end

indexer = cell(1, dim + 1); % Indexer for higher dimensional x, z, u.

% Populate indexer with the appropriate number of : operators.
for i = 1:dim
    indexer{i} = ':';
end

% Perform up to the maximum iterations N steps of the ADMM algorithm. Early
% termination when convergence is reached.
for i = 1:N

    zprev = z;                                  % Previous step's z value.
    indexer{dim + 1} = i;                       % Accessor.
    
    if (alg == 0)
        x = xminf(x, z, u, rho);                % x-Minimization step.
    else
        aprev = acurr;                          % New predictor/corrector.
        uprev = u;                              % Previous step's uhat.
        x = xminf(x, v, uhat, rho);             % x-Minimization step.
        
        if (alg == 2)
            dprev = d;                          % Previous restart value.
        end
    end
    
    % Check if the user specified a relaxation parameter and decide whether
    % to perform relaxation or not.
    if (relax ~= 1)                             % Case of relaxation.
        % Relaxation on A*x.
        Axhat = relax*A(x) - (1 - relax)*(B(zprev) - c);
        
        % Update z based on the algorithm type.
        if (alg == 0)
            z = zming(Axhat, z, u, rho);        % z-Minimization step.
        else
            z = zming(Axhat, z, uhat, rho);     % z-Minimization step.
        end
    else                                        % Case of no relaxation.
        % Update z based on the algorithm type.
        if (alg == 0)
            z = zming(x, z, u, rho);            % z-Minimization step.
        else
            z = zming(x, z, uhat, rho);         % z-Minimization step.
        end
    end
    
    % Store matrix-vector products for updated x and z.
    Ax = A(x);
    Bz = B(z);
    
    if ~isfield(options, 'altu')
        % Update u based on whether or not relaxation is being used.
        if (relax ~= 1)
            if (alg == 0)
                u = u + (Axhat + Bz - c);           % u-Minimization step.
            else
                u = uhat + (Axhat + Bz - c);        % u-Minimization step.
            end
        else
            if (alg == 0)
                u = u + (Ax + Bz - c);              % u-Minimization step.
            else
                u = uhat + (Ax + Bz - c);           % u-Minimization step.
            end
        end
    else
        % u-Minimization over a user defined u-update.
        if (relax ~= 1)
            u = options.altu(u, Axhat, Bz, c);
        else
            u = options.altu(u, Ax, Bz, c);
        end
    end
    
    % Updates of Fast / Accelerated ADMM variables.
    if (alg == 1 || alg == 2)
        if (alg == 1)                           % Case of Fast ADMM.
            % Update predictor / corrector scalar alpha, and consequently
            % compute new predicting x (i.e., v) and predicting u (uhat).
            acurr = 1/2*(1 + sqrt(1 + 4*aprev^2));
            v = z + (aprev - 1)/acurr*(z - zprev);
            uhat = u + (aprev - 1)/acurr*(u - uprev);
        else                                    % Case of Accelerated ADMM.
            % Update the residual norm d-value for Accelerated ADMM.
            d = 1/rho*norm(u - uhat, 'fro')^2 + ...
                rho*norm(B(z - v), 'fro')^2;
            
            % Decide whether to restart or not.
            if (d < n*dprev)                    % Case of not restarting.
                % Update everything as in the case of fast ADMM.
                acurr = 1/2*(1 + sqrt(1 + 4*aprev^2));
                v = z + (aprev - 1)/acurr*(z - zprev);
                uhat = u + (aprev - 1)/acurr*(u - uprev);
                
                results.restarted(i) = 0;       % Report a non-restart.
            else                                % Case of restarting.
                % Restart all Acceleration variables.
                acurr = 1;
                v = zprev;
                uhat = uprev;
                d = dprev/n;
                
                results.restarted(i) = 1;       % Report a restart.
            end
            
            results.dvals(i) = d;               % Record the d-value.
        end
        
        % Store Fast / Accelerated results.
        results.vvals(indexer{:}) = v;
        results.uhatvals(indexer{:}) = uhat;
        results.avals(i) = acurr;
    end
    
    % Evaluate the objective, if possible.
    if canEvalObj
        results.objevals(i) = obj(x, z);        % Current objective value.
    end
    
    % Record this iterations x, z, and u values.
    results.xvals(indexer{:}) = x;
    results.zvals(indexer{:}) = z;
    results.uvals(indexer{:}) = u;
    
    if isfield(options, 'specialnorms') && ...
        isa(options.specialnorms, 'function_handle')
        v = options.specialnorms(x,z,u,rho);
        results.pnorm(i) = v(1);
        results.dnorm(i) = v(2);
    else
        % Compute Primal and Dual norms.
        switch(alg)
            case 0                                  % Case of normal ADMM.
                results.pnorm(i) = norm(Ax + Bz - c, 'fro');

                if ~nodualerror
                    results.dnorm(i) = norm(rho*At(B(z - zprev)), 'fro');
                else
                    results.dnorm(i) = NaN;
                end
            case 1                                  % Case of Fast ADMM.
                results.pnorm(i) = norm(Ax + Bz - c, 'fro');

                if ~nodualerror
                    results.dnorm(i) = rho*norm(At(B(z - v)), 'fro');
                else
                    results.dnorm(i) = NaN;
                end
        end
    end
    
    % Compute Primal and Dual errors, when there's no Accelerated ADMM.
    if (alg == 0 || alg == 1)
        % The number of elements in Ax and Bz. These could differ as the
        % vectors may be different lengths, or the products Ax and Bz might
        % be matrices themselves.
        M1 = numel(Ax);
        M2 = numel(Bz);
        
        % Compute Primal and Dual errors for tolerances RELTOL and ABSTOL.
        results.perr(i) = sqrt(M1)*ABSTOL + ...
            RELTOL*max(max(norm(Ax, 'fro'), norm(Bz, 'fro')), ...
            norm(c, 'fro'));
        
        if ~nodualerror
            results.derr(i) = sqrt(M2)*ABSTOL + ...
                RELTOL*norm(rho*At(u), 'fro');
        else
            results.derr(i) = NaN;
        end
    end
    
    % If output text allowed, report results for non-Accelerated ADMM.
    if (~quiet && alg ~= 2)
        if canEvalObj
            % Print iteration's results.
            fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', i, ...
                results.pnorm(i), results.perr(i), ...
                results.dnorm(i), results.derr(i), results.objevals(i));
        else
            % Print iteration's results.
            fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', i, ...
                results.pnorm(i), results.perr(i), ...
                results.dnorm(i), results.derr(i));
        end
    end
    
    % Convergence testing, when allowed. 
    if (convtest || strcmp(stopcond, 'hnorm') || strcmp(stopcond, 'both'))
        wprev = w;                              % Record previous w.
        w = [x; z; rho*u];                      % Update w.
        
        % Record results for this w and its H-norm squared value.
        results.wvals(indexer{:}) = w;
        results.Hnormsq(i) = H_norm_sq(wprev - w);
        
        % Convergence testing: we check if the H-norm squared values are
        % monotonically decreasing, implying some convergence.
        if (convtest && i >= 2)
            H2 = results.Hnormsq(i);            % Current H-norm squared.
            H1 = results.Hnormsq(i - 1);        % Previous one.
            
            % Check if the relative difference between last two H-norm
            % values satisfies the convergence tolerance.
            if alg == 0 && H1 > eps && H2 > H1 &&...
                ~((H2 - H1) <= H1*convtol)
                fprintf(['Iteration %i: H norms not converging to ', ... 
                    'given relative tolerance: %d is not less or', ...
                    ' equal to tol. %d\n'], i, (H2 - H1)/(H1 + eps), ...
                    convtol);
                disp(['ADMM seems to not be converging! Please ', ...
                    'check that your proximal operators are correct!']);
                return;
            end
        end
    end
    
    % Check stopping conditions. Stop if reached.
    if (alg == 2 && i >= 2 && abs(d-dprev) <= DVALTOL*dprev)
            break;                              % Terminate (Accelerated)
    elseif (alg == 0 || alg == 1)               % Otherwise...
        % Check if the standard stopping condition has been reached.
        if (strcmp(stopcond, 'standard') || strcmp(stopcond, 'both')) &&...
            (~domaxiters && results.pnorm(i) < results.perr(i) && ...
            (nodualerror || results.dnorm(i) < results.derr(i)))
            break;                              % Terminate.
        end
    end
    
    % Check if the H-norm values have reached an absolute difference
    % tolerance HNORMTOL, stopping if true. Only if allowed by user.
    if (strcmp(stopcond, 'hnorm') || strcmp(stopcond, 'both')) && ...
        ~domaxiters && i > 2 && results.Hnormsq(i) <= HNORMTOL
        break;                                  % Terminate.
    end
    
    if adaptive && convtest && i > 2
        growthtol = 5;
        wdiff = H1 - H2;
        rhoprev = rho;
        rho = rho*(wdiff'*rhoprev)/(wdiff'*wdiff);
        %t_adapt =   max(t_adapt*(rprev'*rprev)/(rprev'*rdiff),1e-5);
        
        % Change in step-size.
        rhodiff = abs(rho - rhoprev);

        % If the relative growth/shrinkage of the new step size is too
        % large/small, use a different step size.
        if rhodiff >= rhoprev*growthtol
            rho = rho/growthtol;
        elseif rhodiff <= rhoprev/growthtol
            rho = rho*growthtol;
        end
    end

end

% Record number of iterations to convergence and convergent value of obj.
results.steps = i;
results.xopt = x;
results.zopt = z;
results.uopt = u;

% Evaluate the optimal objective value as computed by ADMM.
if (objevals && isa(obj,'function_handle'))
    results.objopt = obj(x, z);
end

results.runtime = toc(start);                   % Record run time.

% Check if output text is allowed and output results if it is.
if ~quiet
    % Display the runtime.
    fprintf('Elapsed time is %d seconds.', results.runtime);
    
    % Display number of iterations required to converge.
    fprintf('Number of steps to convergence: %d', results.steps);
end

results.options = options;

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Sets an option for ADMM based on provided options struct, with a default
% value to default to.

function output = setopt(struct, opttext, default)
% INPUTS ------------------------------------------------------------------
% struct:   The options struct to set options based off of.
% opttext:  A string of the name of the option to set in the struct.
% default:  The default value of the option.
% 
% OUTPUTS -----------------------------------------------------------------
% output:   The resulting value of the option being set.
% -------------------------------------------------------------------------

output = 0;                 % Set output to 0 initially.

% First of all, check if the field opttext even exists in the struct.
if isfield(struct, opttext)
    % If it does, assign the value from it as output:
    switch(opttext)
        % Informs ADMM to perform parallelized updates on one of the
        % proximal operators xminf and zming. Doing this also causes
        % parallelization of u-update step, for further efficiency. ADMM
        % can handle parallelization of both xminf and zming. Note that
        % when parallel mode is activated, the function being parallelized
        % must accept another parameter i denoting which slice is being
        % minimized. If xminf is being minimized, then parallel xminf has
        % function signature xminf(x,z,u,rho,i). One can control the size
        % of slices by setting options.slices. See this option for more
        % details on slicing. Possible values for parallel:
        %    'xminf'       --> Parallelize proximal operator updates for
        %                      function f(x).
        %    'zming'       --> Parallelize proximal operator updates for
        %                      function g(z).
        %    'both'        --> Parallelize both proximal operators.
        %    'none'        --> Do not perform any parallel updates.
        % The default value is 'none'.
        case 'parallel'
            output = struct.parallel;
        % A vector containing the integer sizes of the slices in parallel
        % ADMM. If the size of each slice should be constant, slices can
        % just be an integer. Setting slice to 1 will perform parallel ADMM
        % vector component-wise. If the slice value is set to 0, ADMM will
        % see how many workers are available and distribute the workload to
        % them as evenly as possible. Default value is 0. Note that if you
        % are performing parallel ADMM on both proximal operators, this
        % input should be a cell where slices{1} corresponds to slices for
        % proximal operator for f, xminf, and slices{2} corresponds to
        % slices for proximal operator for g, xming.
        case 'slices'
            output = struct.slices;
        % Specifies whether to use fast ADMM algorithms or not. This is a
        % binary value with default as 0 (don't use fast). Note that, in
        % general, fast ADMM will have different stopping conditions than
        % normal ADMM: it uses dvaltol below instead of H-norms or the
        % standard stopping condition. If your problem is strongly convex,
        % you can set fasttype below to 'strong' to maintain the same
        % stopping conditions as regular ADMM. 
        case 'fast'
            output = struct.fast;
        % Specifies what type of fast algorithm to user, if the user set
        % the algorithm option to 'fast'. Possible values are:
        %    'weak'     --> Only have weak convexity for either function f
        %                   or g, or both, so use Accelerated ADMM.
        %    'strong'   --> Have strong convexity for both functions f and
        %                   g, so Fast ADMM algorithm is allowed.
        % Default value is 'weak' (theoretically, one can always do
        % Accelerated ADMM).
        case 'fasttype'
            output = struct.fasttype;
        case 'adaptive'
            output = struct.adaptive;
        % The restart parameter. A value between 0 and 1 specifying when
        % restarting should occur in (restart if d values have not changed
        % more than 1 - restart percent). Default is 0.999, and is the
        % recommended value for the restart parameter.
        case 'restart'
            output = struct.restart;
        % Specifies whether output text of iterations and results is
        % allowed. Binary value with default as 1 (be quiet).
        case 'quiet'
            output = struct.quiet;
        % The step size parameter rho to use for ADMM iterations. In the
        % case of Adaptive ADMM, this is the initial rho. Default value is
        % set to 1.
        case 'rho'
            output = struct.rho;
        % The max number of iterations to perform. Must be positive and an
        % integer. Default value is 1000.
        case 'maxiters'
            output = struct.maxiters;
        % Specifies whether the maximum number of iterations should be done
        % even if convergence has been reached. Binary value with default
        % as 0.
        case 'domaxiters'
            output = struct.domaxiters;
        % The relaxation parameter to use. Setting this to a value other
        % than 1 will cause relaxation to be performed in the z updates for
        % ADMM. Default value is 1 (no relaxation).
        case 'relax'
            output = struct.relax;
        % The objective function handle to evaluate in ADMM. Must accept
        % two parameter x and z and be of the form obj(x, z). Default is to
        % be 0.
        case 'obj'
            output = struct.obj;
        % Specifies whether we are allowed to evaluate the objective each
        % iteration. Binary value with default 0 (don't evaluate).
        case 'objevals'
            output = struct.objevals;
        % Specifies whether we are allowed to test that ADMM is converging.
        % Incurs some overhead, but saves user from running bad proximal
        % operators unnecessarily if enabled. Binary value with default 0
        % (don't test convergence).
        case 'convtest'
            output = struct.convtest;
        % The tolerance to use for absolute difference in H-norm squared
        % values. Must be positive and ideally small. Default value is
        % 1e-10 (should be monotonically smaller to machine precision).
        case 'convtol'
            output = struct.convtol;
        % Specifies the type of stopping condition to use. The possible
        % values one can use are:
        %    'standard' --> Use Boyd's convergence checking of relative
        %                   differences of primal and dual residual errors.
        %    'hnorm'    --> Use the H-norm based stopping condition.
        %    'both'     --> Use both stopping conditions and stop when
        %                   either one is reached.
        % Default is 'standard', with standard primal and dual error set.
        case 'stopcond'
            output = struct.stopcond;
        % Specifies the absolute error parameter in Boyd's stopping
        % condition ('standard') for ADMM. Must be positive and ideally 
        % small - smaller than reltol below. Default value is 1e-5.
        case 'abstol'
            output = struct.abstol;
        % Specifies the relative error parameter in Boyd's stopping
        % condition ('standard') for ADMM. Must be positive and ideally 
        % somewhat small. Default value is 1e-3.
        case 'reltol'
            output = struct.reltol;
        % Specifies whether or not to use the dual for residual error in
        % the standard stopping conditions. A binary value. Depending on
        % the problem (e.g., SVM), ADMM's dual error can be unreasonably
        % slow to converge. This will turn it off and focus on primal only.
        % The default value is 0.
        case 'nodualerror'
            output = struct.nodualerror;
        % Specifies the relative tolerance in H-norm squared values for
        % stopping condition checking using the H-norm test. Must be
        % positive and ideally small. Default value is 1e-6.
        case 'Hnormtol'
            output = struct.Hreltol;
        % Specifies the absolute tolerance in d values for Accelerated
        % ADMM. Must be positive and ideally small. Default value is set to
        % 1e-6.
        case 'dvaltol'
            output = struct.dvaltol;
        % The size of the column vector c in the ADMM constraint. Must be a
        % positive integer value. Default value is 0 for error catching.
        case 'm'
            output = struct.m;
        % The number of columns in the matrix A in the ADMM constraint.
        % Must be a positive integer value. Default value is 0 for error
        % catching.
        case 'nA'
            output = struct.nA;
        % The number of columns in the matrix B in the ADMM constraint.
        % Must be a positive integer value. Default value is 0 for error
        % catching.
        case 'nB'
            output = struct.nB;
        % The initial value of x in the ADMM iteration. Can help with
        % convergence if a rough idea of the solution is known. Must be a
        % vector of length nA. Default value is a vector of all 0's.
        case 'x0'
            output = struct.x0;
        % The initial value of z in the ADMM iteration. Can help with
        % convergence if a rough idea of what it should be is known. Must 
        % be a vector of length nB. Default value is a vector of all 0's.
        case 'z0'
            output = struct.z0;
        % The initial value of Lagrange Multiplier u in the ADMM iteration.
        % Can help with convergence if a rough idea of what it should be is
        % known. Must be a vector of length m. Default value is a vector of
        % all 0's.
        case 'u0'
            output = struct.u0;
    end
else
    % Otherwise, give the default value as no value was specified in the
    % options struct.
    output = default;
end
    
end
% -------------------------------------------------------------------------