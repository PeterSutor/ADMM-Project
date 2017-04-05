% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Solves the Total Variation Minimization (TVM) problem using ADMM. TVM 
% minimizes for x the objective function:
%    obj(x) = 1/2||x - s||_2^2 + lambda*sum_i{|x_{i+1} - x_i|}, 
% where x and s are column vectors of length m. The vector s is a given 
% vector of signals to perform TVM on. The minimized vector x is a denoised
% version of s in the sense of total variation. The constant lambda 
% represents how strictly or loosely to minimize the noise in s.
% 
% The ADMM Augmented Lagrangian here is:
%    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||D*x - z + u||_2)^2 + 
%                   constant(u),
% where f(x) = 1/2||D*x - s||_2^2 and g(z) = lambda*||z||_1 (by 
% substitution). The rho in L_rho is the dual step size parameter. ADMM 
% solves the equivalent problem of minimizing: f(x) + g(z), subject to 
% D*x - z = 0. The matrix D is the difference matrix that performs the 
% z_{i+1} - z_i differences for the vector z. Thus, it is a sparse m by m 
% matrix with the stencil [1 -1] along the diagonal entries.
% 
% ADMM requires proximal operators for minimizing L_rho(x,z,u) for x and z,
% separately - i.e., proximal operators for functions f and g. The proximal
% operators are obtained as follows:
%
% For the proximal operator for f, we begin by noting that the gradient of
% L_rho(x,z,u) is:
%    grad_x(L_rho(x,z,u)) = grad_x(f(x)) + rho*D^T*(D*x - z + u)
%                         = (x - s) + rho*D^T*(D*x - z + u)
% To minimize this, we set it equal to 0 and solve for x, giving:
%    (x - s) + rho*D^T*(D*x - z + u) = 0 
%       <--> (I + rho*D^T*D)*x = s + rho*D^T*(z - u)                    (1)
%       <--> x := (I + rho*D^T*D)^-1*(s + rho*D^T*(z - u))
% Thus, we can solve the system (1) for our minimized x. Note that since I
% and rho*D^T*D are clearly sparse, we can abuse the sparsity using sparse
% matrices.
%
% For the proximal operator for g, note that g's formulation is minimizable
% via soft-thresholding, a proximal mapping technique:
%    z = min_z(||z - v||_2^2 + lambda*||v||_1) 
%      = sign(z)*(|z| - lambda/rho)_+ (non positive parts set to 0)
% This function simply evaluates this for v = u + D*x, and appropriate rho.
% Also note that soft-thresholding typically sets rho = 1, but we move
% according to the step size parameter rho. 
%
% Note that in ADMM, the ratio of lambda to rho represents how strictly or
% loosely to minimize the noise in q.
%
% Check the totalvariation section of function getproxops to see the 
% proximal operators.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the totalvariation function is executed with no inputs, it will
% run totalvariationtest with no inputs. This will generate a random
% problem of size n = 2^7, and test it for correctness, showing the 
% results.

function [results] = totalvariation(s, lambda, options)
% INPUT -------------------------------------------------------------------
% s:        The signal vector in the TVM problem definition.
% lambda:   The parameter lambda in in the TVM problem definition,
%           specifying the contribution of the total variation term.
% options:  A struct containing options customizing the ADMM execution. If
%           no options are provided (an empty or irrelevant struct is given
%           as the options parameter), default settings are applied, as 
%           mentioned in the user manual.
% 
% OUTPUT ------------------------------------------------------------------
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
% totalvariationtest function with no arguments. Returns test results for 
% random data of size n = 2^7.
if nargin == 0
    display(['No arguments detected; running a demo test of the', ...
        ' Total Variation Minimization problem for random data of size',...
        ' n = 2^7:']);
    results = totalvariationtest();
    results.solverruntime = toc;                % End timing.
    
    return;
end

% Check input for errors and return fixed values to account for weird (but
% fixable) input.
[s, lambda, n] = errorcheck(s, lambda, options);

% Sparse diagonal difference matrix. Approximates dx as Dx using forward 
% differentiation (stencil [1, -1] in diagonal with circular boundary 
% conditions, normally).
D = spdiags([ones(n, 1) -ones(n, 1)], 0:1, n, n);

Dt = D';                        % Stores transpose of D.
DtD = Dt*D;                     % Stores the product of D transpose and D.
Id = speye(n);                  % Sparse identity matrix.

% The objective function for TVM.
objective = @(x, z) 1/2*norm(x - s, 'fro')^2 + ...
    lambda*sum(abs(x(2:length(x)) - x(1:length(x)-1)));

% Populate the args struct holding fields for getproxops, which returns our
% proximal operators for functions f and g.
args.Id = Id;                   % Cached identity matrix.
args.D = D;                     % Matrix D.
args.Dt = Dt;                   % Cached transpose of D.
args.DtD = DtD;                 % Cached product D^T*D.
args.s = s;                     % Noisy signal vector s.
args.lambda = lambda;           % The lambda / strictness parameter in 
                                % objective function.

% Get our proximal operators.
[xmin, zmin] = getproxops('TotalVariation', args);

% Populate our constraints and constraint sizes for ADMM in options struct.
options.A = D;                  % Matrix D in constrain D*x - z = 0.
options.At = Dt;                % Transpose of D.
options.B = -1;                 % Simply -1, for -z in constraint.
options.mB = n;                 % As B is constant -I_n, pass row size. 
options.nB = n;                 % As B is constant -I_n, pass column size.
options.c = 0;                  % Constant 0 in constraint D*x - z = 0.
options.m = n;                  % Notify ADMM of length of zero vector c.
%options.algorithm = 'fast';     % This problem is weakly convex, we can use
%options.fasttype = 'weak';      % Accelerated ADMM.
%options.dvaltol = 1e-6;         % Moderate tolerance suffices.
options.obj = objective;        % Pass the objective function.

% Solve the TV problem with our ADMM function and return the results.
results = admm(xmin, zmin, options);
results.solverruntime = toc;    % End timing.

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Simply checks for any user error on same input, stops if error is
% detected. If input is correct, return info needed for problem.

function [s, lambda, n] = errorcheck(s, lambda, options)
% INPUTS ------------------------------------------------------------------
% Same as for the totalvariation function.
% 
% OUTPUTS -----------------------------------------------------------------
% s:        The column vector s (noisy signal) in problem specification.
% lambda:   The lambda / strictness parameter in objective function.
% n:        Length of vector s.
% -------------------------------------------------------------------------


% Check to make sure lambda is a valid value.
if (~isnumeric(lambda) || real(lambda) < 0)
    error('Given lambda parameter is not a nonnegative number!');
else
    lambda = real(lambda);      % Assign lambda.
end

% Check that s is a vector and get its size.
if ~isvector(s)
    error('Argument s is not a vector!');
else
    if isrow(s)                 % If not a column vector, make it one.
        s = s';
    end
    
    % Get size of vector s.
    [n, ~] = size(s);
end

% Check that the given options is actually a struct...
if ~isstruct(options)
    error('Given options is not a struct! At least pass empty struct!');
end

end
% -------------------------------------------------------------------------