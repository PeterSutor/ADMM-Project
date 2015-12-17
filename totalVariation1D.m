% Minimize 1/2||x - signal||_2^2 + lambda*sum_i ||x_{i+1} - x_i||_1

function [objvalA, xA, zA, iterA, objvalS, xS, zS, iterS] = ...
    totalVariation1D(signal, lambda, rho, alpha, M, quiet)

% Performs the generic Alternating Direction Method of Multipliers (ADMM)
%   on given input, using scaled dual variable. Solves the Total Variation 
%   Minimization problem.
% -------------------------------------------------------------------------
% INPUT:
% rho           Step size in ADMM.
% b             Noisy input vector to denoise.
% lambda        Regularization parameter in TV problem.
% -------------------------------------------------------------------------
% OUTPUT:
% x             Solution to min(E(x, b) + lambda*V(x)) subject to dx = z;
%               thus the problem is actually min(E(x, b) + lambda*||z||_1),
%               subject to Dx - z = 0 in ADMM form.
% iter          Contains records of the following information for each
%               iteration.
% iter.objval   Vector containing objective evaluations of obj for each
%               iteration.
% iter.pnorm    The primal norm for each iteration.
% iter.dnorm    The dual norm for each iteration.
% iter.perr     The measured primal error based on tolerances ABSTOL and
%               RELTOL for each iteration.
% iter.derr     The measured dual error based on tolerances ABSTOL and
%               RELTOL for each iteration.
% iter.steps    Number of steps needed to converge; value between 1 and N.
% -------------------------------------------------------------------------

n = length(signal);                  % Length of our noisy vector signal.

% Sparse diagonal difference matrix.
%   Approximates dx as Dx using forward differentiation (stencil [1, -1] in
%   diagonal with circular boundary conditions, normally).
D = spdiags([ones(n, 1) -ones(n, 1)], 0:1, n, n);

Dt = D';                        % Stores transpose of D.
DtD = Dt*D;                     % Stores the product of D transpose and D.
Id = speye(n);                  % Sparse identity matrix.

% Forms the functions to pass to Generic ADMM. Function objective is the
%   Total Variation Minimization problem written in ADMM form. Function
%   prox_fx is the proximal operator for x-minimization - is just an
%   analytic solution in this case as E(x, b) is differentiable. Function
%   prox_gz is the proximal operator for z-minimization. Function g is
%   clearly just a scalar times the 1-norm of z, so we use
%   soft-thresholding.
%objective = @(x, z) 0.5*norm(x - signal)^2 + lambda*norm(z, 1);
objective = @(x, z) 1/2*norm(x - signal, 2)^2 + ...
                    lambda*sum(abs(x(2:length(x)) - x(1:length(x)-1)));

args.Id = Id;
args.D = D;
args.Dt = Dt;
args.DtD = DtD;
args.signal = signal;
args.alpha = alpha;
args.lambda = lambda;
[xmin, zmin] = getProxOps('TotalVariation1D', args);

% Solve the TV problem with our ADMM function.
[objvalS, xS, zS, iterS] = admm(objective, xmin, zmin, rho, D, Dt, -Id, ...
    zeros(n, 1), M, quiet);
ddf = @(x, z, u, rho) x - signal + rho*Dt*(D*x - z + u);
[objvalA, xA, zA, iterA] = adaptive_admm(objective, xmin, zmin, ddf, ...
    rho, D, Dt, -Id, zeros(n, 1), M, quiet);

end

