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

function [objval, x, z, iter] = ... 
    adaptive_admm(obj, xminf, zming, ddf, rho, A, At, B, c, N, quiet)
% INPUTS:
% obj       Objective function handle of inputs x and z, respectively.
% xminf     Function handle of inputs x, z and u, respectively. Returns the
%           minimizing x value of the augmented Lagrangian.
% zming     Function handle of inputs x, z and u, respectively. Returns the
%           minimizing z value of the augmented Lagrangian.
% rho       Step size parameter, chosen by user.
% A         An m by n matrix, or function handle of x returning the result
%           of the product Ax.
% At        An n by m matrix that is the transpose of A, A^T.
%           Alternatively, a function handle of x returning the result of
%           the product (A^T)x.
% B         An m by n matrix, or function handle of z returning the result
%           of the product Bz.
% c         A row vector of size n.
% N         Maximum number of iterations to perform barring convergence.
% quiet     A value of 0 specifies to record the execution time, show the
%           contents of iter(i) every iteration i and the number of steps 
%           required to converge. A value of 1 specifies not to do this.
% -------------------------------------------------------------------------
% OUTPUTS: 
% objval        Evaluation of the objective at minimization points x and z.
% x             Minimization point of function obj for the x variable.
% z             Minimization point of function obj for the z variable.
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


if ~quiet
    start = tic;                                % Start timing execution.
end

ABSTOL = 1e-4;                                  % Absolute tolerance.
RELTOL = 1e-3;                                  % Relative tolerance.

m = length(c);                                  % Length of vector c.
x = zeros(m, 1);                                % Initialize x to zero.
z = zeros(m, 1);                                % Initialize z to zero.
u = zeros(m, 1);                                % Initialize u to zero.

% Check if A is a function handle. If not, make it one to simplify cases.
if isnumeric(A)
    A = @(v) A*v;
end

% Check if A^T is a function handle. If not, make it one to simplify cases.
if isnumeric(At)
    At = @(v) At*v;
end

% Check if B is a function handle. If not, make it one to simplify cases.
if isnumeric(B)
    B = @(v) B*v;
end

if ~quiet
    % Print headers for iter's contents.
    fprintf('%7s\t%20s\t%20s\t%20s\t%20s\t%20s\n', 'Iteration', ...
      'Primal Residual Norm', 'Primal Error', 'Dual Residual Norm', ... 
      'Dual Error', 'Objective Value');
end

% Derivative of Dual for f from Esser's
r1 = ddf(x, z, u, rho) + B(z);
minrho = rho;
start = rho;

% Perform up to the maximum iterations N steps of the ADMM algorithm. Early
% termination when convergence is reached.
for i = 1:N
    
    zprev = z;                                  % Previous step's z value.
    
    x = xminf(x, z, u, rho);                    % x-Minimization step.
    z = zming(x, z, u, rho);                    % z-Minimization step.
    
    % Store matrix-vector products for updated x and z.
    Ax = A(x);
    Bz = B(z);
    
    u = u + (Ax + Bz - c);                      % u-Minimization step.
    
    % Populate iter for iteration i.
    % ---------------------------------------------------------------------
    iter.objval(i) = obj(x, z);                    % Current objective value.
    
    % Compute Primal and Dual norms.
    iter.pnorm(i) = norm(Ax + Bz - c);
    iter.dnorm(i) = norm(rho*At(B(z - zprev)));
    
    % Compute Primal and Dual errors for tolerances RELTOL and ABSTOL.
    iter.perr(i) = sqrt(m)*ABSTOL + ...
        RELTOL*max(max(norm(Ax), norm(Bz)), norm(c));
    iter.derr(i) = sqrt(m)*ABSTOL + RELTOL*norm(rho*At(u));
    % ---------------------------------------------------------------------
    
    if ~quiet
        % Print iteration's results.
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', i, ...
            history.pnorm(i), history.perr(i), ...
            history.dnorm(i), history.derr(i), history.objval(i));
    end
    
    % Check stopping condition. Stop if reached.
    if (iter.pnorm(i) < iter.perr(i)) && (iter.dnorm(i) < iter.derr(i))
        break;
    end
    
    % Residual interpolation.
    r0 = r1;
    r1 = ddf(x, z, u, rho) + Bz;
    rdiff = r0 - r1;
    rdifft = rdiff';
    
    %maxrho = max(rho, maxrho);
    rho = abs(-(rho*rdifft*r0)/(rdifft*rdiff));
    minrho = min(rho, minrho);
    
    % Handle errors for rho.
    if (rho == Inf)
        rho = start;
        display('Use max rho for below iteration... ');
    end
    
    disp(['Iteration ', num2str(i), ': rho = ', num2str(rho)]);
    
end

% Record number of iterations to convergence and convergent value of obj.
iter.steps = i;
objval = obj(x, z);

if ~quiet
    toc(start);                                 % Show execution time.
    
    % Display number of iterations required to converge.
    fprintf('Number of steps to convergence: %d', iter.steps);
end

end

