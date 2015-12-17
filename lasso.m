% Lasso Problem with ADMM
% Performs the generic Alternating Direction Method of Multipliers (ADMM)
%   on given input, using scaled dual variable. Solves the Lasso problem.

function [results] = lasso(A, b, lambda, options)
% Performs the generic Alternating Direction Method of Multipliers (ADMM)
%   on given input, using scaled dual variable. Solves the Lasso problem.
% -------------------------------------------------------------------------
% INPUT:
% A             The m by n matrix in the Lasso problem.
% b             n = length(b) vector of data in the Lasso problem.
% lambda        Regularization parameter Lasso problem.
% options       Options argument for ADMM.
% -------------------------------------------------------------------------

Atb = A'*b;                     % Stores product of transpose of A times b.
[m, n] = size(A);               % Gets the dimensions of our problem.
Id_n = speye(n);                % Stores the sparse n by n identity matrix.
Id_m = speye(m);                % Stores the sparse m by m identity matrix.
%--------------------------------------------------------------------------

if (isfield(options, 'rho'))
    rho = options.rho;
else
    rho = 1.0;
end

if (isfield(options, 'relax'))
    relax = options.relax;
else
    relax = 1.0;
end

% Get an LU decomposition for the x-minimization step.
if(m >= n)                      % A is square or tall.
    % Get lower triangular L for traspose of A times A plus rho for n.
    L = chol(A'*A + rho*Id_n, 'lower');
else                            % A is short and fat.
    % Instead, scale by rho and swap roles of A and A transpose. Need to
    %   use m by m identity matrix in this case.
    L = chol(1/rho*(A*A') + Id_m, 'lower');
end

L = sparse(L);                  % Sparse version of our L.
U = sparse(L');                 % Upper triangular U is trivial; simply the
                                %   transpose of L. Made sparse.

args.A = A;
args.Atb = Atb;                             
args.L = L;
args.U = U;
args.m = m;
args.n = n;
args.alpha = relax;
args.lambda = lambda;
[minx, minz] = getProxOps('LASSO', args);

obj = @(x, z) 0.5*sum((A*x - b).^2) + lambda*norm(z, 1);

options.obj = obj;
options.A = eye(n, n);
options.At = options.A';
options.B = -options.A;
options.c = zeros(n, 1);

% Solve the TV problem with our GenericADMM function.
[results] = admm(minx, minz, options);

end
