% Coded by:     Peter Sutor Jr.
% Last edit:    5/11/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% An example demonstrating the differences in convergence between regular
% ADMM, Accelerated ADMM, and Fast ADMM, on the Model problem. The Model
% problem is strongly convex, therefore we can demonstrate Fast ADMM on it.
%
% See function model to learn more about the Model problem and function
% modeltest to learn more about testing it.
%
% NOTE: If this function is executed with no inputs, m = n = 100 is used.

function fasteradmmcomparison(m, n, dvaltol, restart)
% INPUTS ------------------------------------------------------------------
% m:        The number of rows in the Model problem's dimensions.
% n:        The number of columns in the Model problem's dimensions.
% dvaltol:  The tolerance to use for Accelerated ADMM's stopping condition.
% restart:  The restart parameter used in Accelerated ADMM.
% 
% OUTPUTS -----------------------------------------------------------------
% None, apart from plots.
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

% If no arguments given, run an example.
if nargin == 0
    m = 200;
    n = 200;
    dvaltol = 1e-8;
    restart = 0.999;
end

rng(0);                                             % Constant RNG seed.

% Generate normally distributed, random data for the model test.
P = randn(m, n);
Q = randn(m, n);
r = randn(n, 1);
s = randn(n, 1);

% Dual step length rho to use.
rho = 1.0;

% Options struct for vanilla ADMM. Use both stopping conditions.
optionsV.stopcond = 'both';
optionsV.rho = rho;

% Options struct for Accelerated ADMM.
optionsA = optionsV;
optionsA.fast = 1;
optionsA.dvaltol = dvaltol;
optionsA.restart = restart;

% Options struct for Fast ADMM (specify strong convexity).
optionsF = optionsA;
optionsF.fasttype = 'strong';

% Run the model problem normally using the model problem solver, for each
% type of ADMM.
resultsV = model(P, Q, r, s, optionsV);
resultsA = model(P, Q, r, s, optionsA);
resultsF = model(P, Q, r, s, optionsF);

% Figure of H-norm-squared residuals. Encodes ADMM's results for x, z, and
% u as column vector w = [x z u]^T and uses a special matrix norm defined
% by matrix H. Shows these for all three variants of ADMM.
figure
semilogy(1:length(resultsV.Hnormsq), max(1e-8, resultsV.Hnormsq), 'r', ...
    1:length(resultsA.Hnormsq), max(1e-8, resultsA.Hnormsq), 'b', ...
    1:length(resultsF.Hnormsq), max(1e-8, resultsF.Hnormsq), 'g', ...
    'LineWidth', 2);
title('Plot of H-Norm Squared Residuals (w = [x^T z^T u^T]^T)');
legend('Vanilla ADMM', 'Accelerated ADMM', 'Fast ADMM', 'Location', ...
    'northeast');
ylabel('||w^k - w^{k+1}||_H^2');
xlabel('Iteration k');

% Figure of Accelerated ADMM's special norm.
figure
semilogy(1:length(resultsA.dvals), max(1e-8, resultsA.dvals), 'b', ...
    'LineWidth', 2);
title('Plot of Accelerated Residual Norms');
legend('Accelerated ADMM', 'Location', 'northeast');
ylabel('1/\rho||u^k - u_{hat}^k||^2 + \rho||B(z^k - z_{hat}^k)||^2');
xlabel('Iteration k');

figure

% Subplot figure of the Primal residual norms, for all three variants of
% ADMM.
subplot(2, 1, 1);
semilogy(1:length(resultsV.pnorm), max(1e-8, resultsV.pnorm), 'r', ...
    1:length(resultsF.pnorm), max(1e-8, resultsF.pnorm), 'g', ...
    'LineWidth', 2);
title('Plot of Primal Residual Norms');
legend('Vanilla ADMM', 'Fast ADMM', 'Location', ...
    'northeast');
ylabel('||Ax^k - Bz^k - c||_2');

% Subplot figure of the Dual residual norms, for all three variants of
% ADMM.
subplot(2, 1, 2);
semilogy(1:length(resultsV.dnorm), max(1e-8, resultsV.dnorm), 'r', ...
    1:length(resultsF.dnorm), max(1e-8, resultsF.dnorm), 'g', ...
    'LineWidth', 2);
title('Plot of Dual Residual Norms');
legend('Vanilla ADMM', 'Fast ADMM', 'Location', ...
    'northeast');
ylabel('||\rho*A^T*B*(z^k-z^{k-1})||_2');

end
% -------------------------------------------------------------------------