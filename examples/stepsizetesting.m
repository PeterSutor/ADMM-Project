% Coded by:     Peter Sutor Jr.
% Last edit:    5/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Performs tests of various step sizes specified by a vector rho, and
% record the convergence of H-norm squared values and prima/dual errors on
% plots, for tester function of a particular solver. The intended use of
% this function is to study the effect of rho on the convergence, or
% non-convergence, of various types of problems.

function results = stepsizetesting(rhos, tester, options)
% INPUTS ------------------------------------------------------------------
% rhos:     A vector containing the values of rho to try.
% tester:   A function handle accepting an options struct. Everything else
%           should be set up to run the tester function this function 
%           handle will execute. E.g.:
%                  tester = @(opts) lassotest(0,100,20,0.001,opts)
% options:  An options struct specifying execution of the tester.
%
% OUTPUTS -----------------------------------------------------------------
% results:  Holds the results struct of every test for every rho value in
%           vector rhos, from the first to last element, in a cell array.
%
% This function also generates plots of the H-norm squared values and
% primal/dual errors for every rho provided.
% -------------------------------------------------------------------------

% Persistent global variable to indicate whether paths have been set or
% not.
global setup;

% Check if paths need to be set up.
if isempty(setup)
    currpath = pwd;                         % Save current directory.
    
    % Get directory of this function (assumed to be in testers folder of 
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

% Number of rhos to try, smooth color transitions to make the lines stand
% out on the plot, set the stopping condition for ADMM to utilize both
% stopping conditions, and two cell arrays intended to hold entries for
% legends of the plots and to hold results of execution.
n = length(rhos);
colors = 1/n:1/n:1;
options.stopcond = 'both';
options.convtest = 0;
legendentries = cell(n, 1);
results  = cell(n, 1);

% Loop to generate tester results for every rho and output the H-norm
% squared values for each run on a plot, with differing colors.
for i = 1:n 
    % Set the next rho to use in the options struct and pass it to the
    % tester function handle provided; record only the results struct.
    options.rho = rhos(i);
    [results{i}, ~] = tester(options);
    
    % Create the plot and generate a legend entry.
    figure(1)
    semilogy(1:results{i}.steps, results{i}.Hnormsq, 'color', ...
        [colors(i), 0, 0], 'LineWidth', 2);
    legendentries{i} = ['rho = ', num2str(rhos(i))];
    
    hold on
end

% Pretty up the graphs and set the legends.
legend(legendentries);
title('Plot of H-Norm Squared Residuals (w = [x^T z^T u^T]^T)');
ylabel('||w^{k - 1} - w^k||_H^2');
xlabel('Iteration k');

% Plot the primal error on a subplot of a different figure.
for i = 1:n    
    figure(2)
    subplot(2, 1, 1);
    
    semilogy(1:results{i}.steps, results{i}.pnorm, 'color', ...
        [0, colors(i), 0], 'LineWidth', 2);
    
    hold on
end

% Set the legend entries for this plot and other things.
legend(legendentries);
title('Plot of Primal Residual Norm');
ylabel('||Ax^k - Bz^k - c||_2');

% Plot the dual errors on the subplot of the prior figure.
for i = 1:n
    figure(2)
    subplot(2, 1, 2);
    semilogy(1:results{i}.steps, results{i}.dnorm, 'color', ...
    [0, 0, colors(i)], 'LineWidth', 2);
    
    hold on
end

% Set the legend entries for this plot and other things.
legend(legendentries);
title('Plot of Dual Residual Norm');
ylabel('||\rho*A^T*B*(z^k-z^{k-1})||_2');
xlabel('Iteration k');

end
% -------------------------------------------------------------------------