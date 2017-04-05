% Coded by:     Peter Sutor Jr.
% Last edit:    5/11/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Creates a small demo showing the effect of relaxation on the Model
% problem. We predict the Model problem to benefit from over-relaxation (a
% relaxation parameter greater than 1) and observe this to be true.
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

m = 150;        % Number of rows in the Model problem.
n = 30;         % Number of columns in the Model problem.

% The relaxation values to try, their colors on the plot, a results cell to
% hold the results structs from running the Model problem for each
% relaxation value, and a cell containing legend entries for each
% relaxation value.
relax = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3];
colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y'];
results = cell(1, 7);
lines = cell(1, 7);

% Run modeltest on the same problem with differing relax parameters.
for i = 1:length(relax)
    % Set the options struct to use the next relaxation parameter. Then,
    % run modeltest and save the results in the results cell. Build our
    % legend entry for this relaxation parameter.
    options.relax = relax(i);
    results{i} = modeltest(0, m, n, 0.001, 1, options);
    lines{i} = ['Relax = ' num2str(relax(i))];
end

% Figure for H-norm squared results.
figure;

% Plot the H-norm squared results for each relaxation parameter.
for i = 1:length(relax)
    semilogy(1:length(results{i}.Hnormsq), ...
    max(1e-8, results{i}.Hnormsq), colors(i),  'LineWidth', 2);
    hold on;
end

% Pretty up the plot.
legend(lines);
xlabel('Iteration k');
ylabel('||w^{k - 1} - w^k||_H^2');
title('H-norm Squared On Model Problem With Differing Relaxations');

% Figure to hold the primal and dual residual results.
figure;
subplot(2, 1, 1);       % Subplot holding the primal results.

% Plot the primal residual norms for each relaxation parameter.
for i = 1:length(relax)
    semilogy(1:length(results{i}.pnorm), ...
    max(1e-8, results{i}.pnorm), colors(i),  'LineWidth', 2);
    hold on;
end

% Pretty up the plot.
legend(lines);
ylabel('||Ax^k - Bz^k - c||_2');
title('Primal Residual Norms On Model Problem With Differing Relaxations');

subplot(2, 1, 2);       % Subplot holding the dual results.

% Plot the dual residual norms for each relaxation parameter.
for i = 1:length(relax)
    semilogy(1:length(results{i}.dnorm), ...
    max(1e-8, results{i}.dnorm), colors(i),  'LineWidth', 2);
    hold on;
end

% Pretty up the plot.
legend(lines);
xlabel('Iteration k');
ylabel('||\rho*A^T*B*(z^k-z^{k-1})||_2');
title('Dual Residual Norms On Model Problem With Differing Relaxations');