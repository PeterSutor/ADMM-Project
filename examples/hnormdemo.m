% Coded by:     Peter Sutor Jr.
% Last edit:    5/11/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Creates a small demo showing the monotonically decreasing H-norm squared
% values in ADMM. See the user manual for more information about the H-norm
% and its encoding of the ADMM iteration in w = [x^T z^T u^T]^T. The demo
% uses the Model problem to generate these values. See function model for
% more information about this solver.
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

% Demo parameters. Change these as desired.
m = 150;                % Number of rows in the Model problem.
n = 30;                 % Number of columns in the Model problem.
errtol = 0.001;         % The relative error tolerance for successful tests

% Set the options struct to use convergence testing and a non-convergence
% tolerance of 10^-16, machine error in single precision.
options.convtest = 1;
options.convtol = 1e-16;

% Create and solve a random Model problem.
[results, test] = modeltest(300, m, n, errtol, 1, options);

% Figure of H-norm-squared residuals. Encodes ADMM's results for x, z, and
% u as column vector w = [x z u]^T and uses a special matrix norm defined
% by matrix H.
figure
semilogy(1:length(results.Hnormsq), max(1e-8, results.Hnormsq), 'k', ...
    1:length(results.Hnormsq), zeros(length(results.Hnormsq), 1) + ...
    results.Hnormtol, 'b--', 'LineWidth', 2);
title('Plot of H-Norm Squared Residuals (w = [x^T z^T u^T]^T)');
legend('Model Problem H-norms squared', 'Convergence Threshold', ...
    'Location', 'northeast');
ylabel('||w^k - w^{k+1}||_H^2');
xlabel('Iteration k');