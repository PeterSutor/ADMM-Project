% Coded by:     Peter Sutor Jr.
% Last edit:    5/11/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% An example demonstrating convergence checking in ADMM, using the Model
% problem solver. Creates a random Model problem of size m by n and tests
% what happens with the convergence checker if either the proximal operator
% for f or for g are incorrect, or both are incorrect, or both are correct.
% Creates a plot of the H-norm squared values used for convergence
% checking for each example and compares them.
%
% See function model to learn more about the Model problem and function
% modeltest to learn more about testing it.
%
% NOTE: If this function is executed with no inputs, m = n = 100 is used.

function convergencechecking(m, n)
% INPUTS ------------------------------------------------------------------
% m:    The number of rows in the Model problem's dimensions.
% n:    The number of columns in the Model problem's dimensions.
% 
% OUTPUTS -----------------------------------------------------------------
% None.
% -------------------------------------------------------------------------


% If no arguments given, run an example.
if nargin == 0
    m = 100;
    n = 100;
end

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

% The number of iterations to perform.
N = 50;

rng(0);                                             % Constant RNG seed.

% Generate normally distributed, random data for the model test.
P = randn(m, n);
Q = randn(m, n);
r = randn(n, 1);
s = randn(n, 1);

% Dual step length rho to use, and previous rho value.
rho = 1.0;
rhoprev = rho;

% Cache some important computations.
Pt = P';
Qt = Q';
PtP = Pt*P;
QtQ = Qt*Q;
Ptr = Pt*r;
Qts = Qt*s;
PtPnew = PtP;
QtQnew = QtQ;

% Efficiently perturb the diagonals of P^T*P and Q^T*Q by rho*I.
PtPnew(1:n+1:end) = PtP(1:n+1:end) + rho;
QtQnew(1:n+1:end) = QtQ(1:n+1:end) + rho;

% Set argument struct needed by getproxops for it to return proximal
% operators for the model problem.
args.PtP = PtP;
args.QtQ = QtQ;
args.Ptr = Ptr;
args.Qts = Qts;
args.n = n;

% Obtain two instances of the Model problems proximal operators. The
% first is for a working proximal operator for g. The second is for f.
% These are used to for the 'one broken prox-op, one correct' tests below.
[~, minz] = getproxops('model', args);
[minx, ~] = getproxops('model', args);

% Set the options struct to configure Model solver execution.
optnormal.convtest = 1;                 % Use convergence testing.
optnormal.convtol = 1e-16;              % Tolerance for non-convergence.
optnormal.maxiters = N;                 % Maximum number of iterations.
optnormal.domaxiters = 1;               % Force ADMM to bypass stopping
                                        % conditions and do maximum number
                                        % of iterations.

% Options and constraint settings for the direct calls to ADMM.
options.convtest = 1;                   % Use convergence testing.
options.convtol = Inf;                  % Tolerance for non-convergence.
options.maxiters = N;                   % Maximum number of iterations.
options.domaxiters = 1;                 % Force ADMM to bypass stopping
                                        % conditions and do maximum number
                                        % of iterations.
options.A = 1;                          % A = I, set to 1 for efficiency.
options.B = -1;                         % B = -I, set to -1 for efficiency.
options.c = 0;                          % c = 0 vector, set to simply 0.
options.m = n;                          % Length of c.
options.nA = n;                         % Number of columns in A.
options.nB = n;                         % Number of rows in B.

% Run the model problem normally using the model problem solver, checking
% for convergence.
normal = model(P, Q, r, s, optnormal);

% Run and obtain results on all three types of broken models. By setting
% the tolerance for non-convergence to infinity in the options struct, the
% algorithms won't prematurely terminate and will show the H-norm squared
% values they compute in the returned results structs.
brokenf = admm(@xminModelBroken, minz, options);
brokeng = admm(minx, @zminModelBroken, options);
broken = admm(@xminModelBroken, @zminModelBroken, options);

options.convtol = 1e-16;                % Set tolerance to realistic level.

% Re-run the prior tests and see if ADMM reports non-convergence as we
% would expect, with a more realistic, machine level tolerance for
% non-convergence.
admm(@xminModelBroken, minz, options);
admm(minx, @zminModelBroken, options);
admm(@xminModelBroken, @zminModelBroken, options);

% Figure of H-norm-squared residuals. Encodes ADMM's results for x, z, and
% u as column vector w = [x z u]^T and uses a special matrix norm defined
% by matrix H.
figure
semilogy(1:length(normal.Hnormsq), max(1e-8, normal.Hnormsq), 'r', ...
    1:length(brokenf.Hnormsq), max(1e-8, brokenf.Hnormsq), 'b', ...
    1:length(brokeng.Hnormsq), max(1e-8, brokeng.Hnormsq), 'g', ...
    1:length(broken.Hnormsq), max(1e-8, broken.Hnormsq), 'm', ...
    'LineWidth', 2);
title('Plot of H-Norm Squared Residuals (w = [x^T z^T u^T]^T)');
legend('Normal Model', 'Model w/ Broken f Prox-op', ...
    'Model w/ Broken g Prox-op', 'Model w/ Both Prox-ops Broken', ...
    'Location', 'northwest');
ylabel('||w^k - w^{k+1}||_H^2');
xlabel('Iteration k');


% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% NOTE: Below are the 'broken' versions of the Model problem's proximal
% operators. By broken, we mean that they should not converge due to a
% change in their code. We point out the incorrect code in the comments.
% Feel free to alter the code in other ways to break it.
% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Evaluates the model problem's (see main function above) proximal operator
% for corresponding function f. The proximal operator is the result of
% taking an x derivative of the model's L_rho and solving it for 0, i.e.:
%    P^T*(P*x - r) + rho*(x - z + u) = 0
% solved for x. This involves solving a simple linear system.

function [minx] = xminModelBroken(~, z, u, rho)
% INPUTS ------------------------------------------------------------------
% x:    The x input corresponding to the function f(x) in the model
%       problem. See main function above for more details. This input is
%       excluded as it is not necessary in the minimization, but is part of
%       the function call and must be included anyway.
% z:    The z input corresponding to the function g(z) in the model
%       problem. See main function above for more details.
% u:    The Lagrange Multiplier variable in ADMM's Augmented Lagrangian
%       L_rho(x, z, u). See main function above for more details.
% rho:  The step size parameter for ADMM.
%
% OUTPUTS -----------------------------------------------------------------
% minx: The minimal x as described in the description.
% -------------------------------------------------------------------------

    if (rho ~= rhoprev)
        % Efficiently add rho to the diagonal entries of P^T*P.
        PtPnew(1:n+1:end) = PtP(1:n+1:end) + rho;
    end
    
    % Solve the linear system from the description for minimal x.
    minx = PtPnew \ (Ptr + rho*(z + u));    % ERROR: should be z - u!
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Evaluates the model problem's (see main function above) proximal operator
% for corresponding function g. The proximal operator is the result of
% taking a z derivative of the model's L_rho and solving it for 0, i.e.:
%    Q^T*(Q*z - s) - rho*(x - z + u) = 0
% solved for z. This involves solving a simple linear system.

function [minz] = zminModelBroken(x, ~, u, rho)
% INPUTS ------------------------------------------------------------------
% x:    The x input corresponding to the function f(x) in the model
%       problem. See main function above for more details. 
% z:    The z input corresponding to the function g(z) in the model
%       problem. See main function above for more details. This input is
%       excluded as it is not necessary in the minimization, but is part of
%       the function call and must be included anyway.
% u:    The Lagrange Multiplier variable in ADMM's Augmented Lagrangian
%       L_rho(x, z, u). See main function above for more details.
% rho:  The step size parameter for ADMM.
%
% OUTPUTS -----------------------------------------------------------------
% minz: The minimal z as described in the description.
% -------------------------------------------------------------------------
    
    if (rho ~= rhoprev)
        % Efficiently add rho to the diagonals of Q^T*Q.
        QtQnew(1:n+1:end) = QtQ(1:n+1:end) + rho;
    end
    
    % Solve the linear system from the description for minimal z.
    minz = QtQnew \ (Qts - rho*(x + u)); % ERROR: Should be + rho
end
% -------------------------------------------------------------------------

end
% -------------------------------------------------------------------------