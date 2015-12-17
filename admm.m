% Coded by:     Peter Sutor Jr.
% Last edit:    6/18/2015
% -------------------------------------------------------------------------
% Description:
%
% Performs the Alternating Direction Method of Multipliers (ADMM) algorithm
% on given input. ADMM minimizes f(x) + g(z) such that Ax + Bz = c, where
% the matrix A is of size m by n, matrix B is of size m by n, and c is a
% row vector of size m. The functions f and g must be convex. This 
% particular ADMM uses scaled dual variables.
%
% Requires given argument minimizing functions of the augmented Lagrangian
% function L_rho(x,z,u) = L_rho(x,z,y) = f(x) + g(z) + y^T(Ax + Bz - c) +
% (rho/2)(||Ax + Bz - c||_2)^2 = f(x) + g(z) + (rho/2)(||Ax + Bz - c + 
% u||_2)^2 + constant, where u = y/rho (the scaled dual variable). One
% function returns the minimizing x, the other z. The constant in 
% L_rho(x,z,u) is unnecesary as it does not affect the minimization points.

function results = admm(xminf, zming, options)

%adaptive = setopt(options, 'useadaptive', options.useadaptive, 0);
%ddf = setopt(options, 'ddf', options.ddf, 0);

quiet = setopt(options, 'quiet', 1);
rho = setopt(options, 'rho', 1.0);
N = setopt(options, 'maxiters', 250);
domaxiters = setopt(options, 'domaxiters', 0);
relax = setopt(options, 'relax', 1);

obj = setopt(options, 'obj', 0);
objevals = setopt(options, 'objevals', 0);

convtest = setopt(options, 'convtest', 0);
convtol = setopt(options, 'convtol', 1e-3);

stopcond = setopt(options, 'stopcond', 'standard');

ABSTOL = setopt(options, 'abstol', 1e-4);
RELTOL = setopt(options, 'reltol', 1e-3);
HNORMTOL = setopt(options, 'Hnormtol', 1e-5);

m = setopt(options, 'm', 0);
nA = setopt(options, 'nA', 0);
nB = setopt(options, 'nB', 0);

if isfield(options, 'c')
    c = options.c;
    
    if (isvector(c))
        if (isrow(c))
            c = c.';
        end
        
        m = length(c);
    else
        error('Given c in constraint Ax + Bz = c is not a vector!');
    end
else
    if (m > 0)
        if (floor(m) == m)
            c = zeros(m, 1);
        else
            error('Noninteger size m of c in constraint Ax + Bz = c!');
        end
    else
        error('Must specify a vector c in constraint Ax + Bz = c!');
    end
end


if isfield(options, 'A')
    A = options.A;
    
    if ismatrix(A)
        [mA, nA] = size(A);
        options.At = A';
        A = @(v) A*v;
    elseif isa(A,'function_handle')
        [mA, nA] = size(A(zeros(nA, 1)));
    else
        error(['Given A in constraint Ax + Bz = c is neither a ', ... 
            'numeric matrix nor function handle of single vector!']);
    end
    
    if (mA ~= m)
        error(['Number of rows in matrix A do not match length of ', ...
            'column vector c in constraint Ax + Bz = c']);
    end
else
    error('Must specify a matrix A in constraint Ax + Bz = c!');
end


if isfield(options, 'At')
    At = options.At;
    
    if ismatrix(At)
        [nAt, mAt] = size(At);
        At = @(v) At*v;
    elseif isa(At,'function_handle')
        [nAt, mAt] = size(At(zeros(mA, 1)));
    else
        error(['Given At (A transpose) in constraint Ax + Bz = c is ', ... 
            'neither a numeric matrix nor function ', ...
            'handle of single vector!']);
    end
    
    if (mAt ~= mA)
        error(['Number of columns in At (A transpose) does not match ', ... 
            'number of rows in A, in constraint Ax + Bz = c']);
    end
    
    if (nAt ~= nA)
        error(['Number of rows in At (A transpose) do not match ', ... 
            'number of columns in A, in constraint Ax + Bz = c']);
    end
else
    error('Must specify a matrix A in constraint Ax + Bz = c!');
end


if isfield(options, 'B')
    B = options.B;
    
    if ismatrix(B)
        [mB, nB] = size(B);
        B = @(v) B*v;
    elseif isa(B,'function_handle')
        [mB, nB] = size(B(zeros(nB, 1)));
    else
        error(['Given B in constraint Ax + Bz = c is neither a ', ... 
            'numeric matrix nor function handle of single vector!']);
    end
    
    if (mB ~= m)
        error(['Number of rows in matrix B do not match length of ', ...
            'column vector c in constraint Ax + Bz = c']);
    end
else
    error('Must specify a matrix B in constraint Ax + Bz = c!');
end

canEvalObj = (objevals && isa(obj, 'function_handle'));

x = setopt(options, 'x0', zeros(nA, 1));
z = setopt(options, 'z0', zeros(nB, 1));
u = setopt(options, 'u0', zeros(m, 1));

results.x0 = x;
results.z0 = z;
results.u0 = u;

if (convtest)
    H_norm_sq = @(wdiff) rho*norm(B(wdiff(nA+1:nA + nB, :)))^2 + ...
        rho*norm(wdiff(nA + nB + 1:nA + nB + m, :))^2;
    w = [x; z; rho*u];
end

start = tic;                                % Start timing execution.

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

% Perform up to the maximum iterations N steps of the ADMM algorithm. Early
% termination when convergence is reached.
for i = 1:N
    
    zprev = z;                                  % Previous step's z value.
    
    x = xminf(x, z, u, rho);                    % x-Minimization step.
    
    if (relax ~= 1)
        xhat = relax*x + (1 - relax)*zold;
        z = zming(xhat, z, u, rho);
        Axhat = A(xhat);
    else
        z = zming(x, z, u, rho);                % z-Minimization step.
    end
    
    % Store matrix-vector products for updated x and z.
    Ax = A(x);
    Bz = B(z);
    
    if (relax ~= 1)
        u = u + (Axhat + Bz - c);               % u-Minimization step.
    else
        u = u + (Ax + Bz - c);                  % u-Minimization step.
    end
    
    % Populate iter for iteration i.
    % ---------------------------------------------------------------------
    if canEvalObj
        results.objevals(i) = obj(x, z);              % Current objective value.
    end
    
    results.xvals(:, i) = x;
    results.zvals(:, i) = z;
    results.uvals(:, i) = u;
    
    % Compute Primal and Dual norms.
    results.pnorm(i) = norm(Ax + Bz - c);
    results.dnorm(i) = norm(rho*At(B(z - zprev)));
    
    % Compute Primal and Dual errors for tolerances RELTOL and ABSTOL.
    results.perr(i) = sqrt(m)*ABSTOL + ...
        RELTOL*max(max(norm(Ax), norm(Bz)), norm(c));
    results.derr(i) = sqrt(m)*ABSTOL + RELTOL*norm(rho*At(u));
    % ---------------------------------------------------------------------
    
    if ~quiet
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
    
    if (convtest || strcmp(stopcond, 'H-norm') || strcmp(stopcond, 'both'))
        wprev = w;
        w = [x; z; rho*u];
        
        results.wvals(:, i) = w;
        results.Hnormsq(i) = H_norm_sq(wprev - w);
        
        if (convtest && i >= 3)
            H2 = results.Hnormsq(i);
            H1 = results.Hnormsq(i - 1);
            
            if H1 > eps && ~((H2 - H1)/H1 <= convtol)
                error(['Iteration %i: H norms not converging to ', ... 
                    'given relative tolerance: %d is not less or equal to tol. %d\n'], ...
                    i, (H2 - H1)/H1, convtol);
            end
        end
    end
    
    % Check stopping condition. Stop if reached.
    if (strcmp(stopcond, 'standard') || strcmp(stopcond, 'both')) && ...
        (~domaxiters && results.pnorm(i) < results.perr(i) && ...
        (results.dnorm(i) < results.derr(i)))
        break;
    end
    
    if (strcmp(stopcond, 'H-norm') || strcmp(stopcond, 'both')) && ...
        ~domaxiters && results.Hnormsq(i) <= HNORMTOL
        break;
    end
end

% Record number of iterations to convergence and convergent value of obj.
results.steps = i;
results.xopt = x;
results.zopt = z;
results.uopt = u;

if (isa(obj,'function_handle'))
    results.objopt = obj(x, z);
end

results.runtime = toc(start);

if ~quiet
    fprintf('Elapsed time is %d seconds.', results.runtime);
    
    % Display number of iterations required to converge.
    fprintf('Number of steps to convergence: %d', results.steps);
end

end


function output = setopt(struct, opttext, default)

output = 0;

if isfield(struct, opttext)
    switch(opttext)
        case 'adaptive'
            output = struct.adaptive;
        case 'ddf'
            output = struct.ddf;
        case 'quiet'
            output = struct.quiet;
        case 'rho'
            output = struct.rho;
        case 'maxiters'
            output = struct.maxiters;
        case 'domaxiters'
            output = struct.domaxiters;
        case 'relax'
            output = struct.relax;
        case 'obj'
            output = struct.obj;
        case 'objevals'
            output = struct.objevals;
        case 'convtest'
            output = struct.convtest;
        case 'convtol'
            output = struct.convtol;
        case 'stopcond'
            output = struct.stopcond;
        case 'abstol'
            output = struct.abstol;
        case 'reltol'
            output = struct.reltol;
        case 'Hnormtol'
            output = struct.Hreltol;
        case 'm'
            output = struct.m;
        case 'nA'
            output = struct.nA;
        case 'nB'
            output = struct.nB;
        case 'x0'
            output = struct.x0;
        case 'z0'
            output = struct.z0;
        case 'u0'
            output = struct.u0;
    end
else
    output = default;
end
    
end

