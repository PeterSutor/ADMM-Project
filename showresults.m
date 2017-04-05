% Coded by:     Peter Sutor Jr.
% Last edit:    5/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Shows pertinent results about a test of a solver, including convergence
% plots and general information about the execution. See the code below for
% all the possible outputs that can be shown. Generally, the results struct
% should be passed as is. The test struct is generally the output of a
% tester, see the user manual for details about the information contained
% in such a struct.

function showresults(results, test, options)
% INPUTS ------------------------------------------------------------------
% results:  A struct containing the results returned by ADMM's execution.
% test:     A struct containing fields for a test problem and its results.
% options:  A struct containing additional information.
%
% OUTPUTS -----------------------------------------------------------------
% None, apart from text output to the interpreter and plots.
% -------------------------------------------------------------------------


% Handle varying input lengths.
if nargin == 0
    error('No structs given to show results for! No arguments given.');
elseif nargin == 1
    test = struct();
    options = struct();
elseif nargin == 2
    options = struct();
end

disp(' ');                      % Some whitespace before results.

% If a solver is provided, write out its problem type. Otherwise, go with
% 'ADMM' as solver.
if isfield(options, 'solver')
    % Obtain lower case of input.
    options.solver = lower(options.solver);
    
    % Switch statement to decide the header.
    switch(options.solver)
        case 'model'
            solver = 'MODEL';
        case 'basispursuit'
            solver = 'BASIS PURSUIT';
        case 'covarianceselection'
            solver = 'SPARSE INVERSE COVARIANCE SELECTION';
        case 'huberfit'
            solver = 'HUBER FITTING';
        case 'lad'
            solver = 'LEAST ABSOLUTE DEVIATIONS';
        case 'lasso'
            solver = 'LASSO';
        case 'linearprogram'
            solver = 'LINEAR PROGRAMMING';
        case 'linearsvm'
            solver = 'LINEAR SUPPORT VECTOR MACHINE';
        case 'quadraticprogram'
            solver = 'QUADRATIC PROGRAMMING';
        case 'totalvariation'
            solver = 'TOTAL VARIATION MINIMIZATION';
        case 'unwrappedadmm'
            % What to do here?
        otherwise
            solver = 'ADMM';
    end
else
    solver = 'ADMM';
end

% Case of a loss function being provided, output an approriate header
% including it. Else, just output the name normally.
if isfield(options, 'lossfunction')
    disp([solver, ' EXECUTION AND TEST RESULTS FOR ', ...
        upper(options.lossfunction), ' ---'])
else
    disp([solver, ' EXECUTION AND TEST RESULTS ---']);
end

% The true optimal object value.
if isfield(test, 'trueobjopt')
    disp(['True optimal objective value: ', num2str(test.trueobjopt)]);
end

% An objective value created by a test to is to be minimized.
if isfield(test, 'testobj')
    disp(['Test''s original objective value: ', num2str(test.testobj)]);
end

% An objective value created by a test using only x.
if isfield(test, 'testobjx')
    disp(['Test''s original objective value for x: ', ...
        num2str(test.testobj)]);
end

% The optimal objective using only x.
if isfield(test, 'objoptx')
    disp(['ADMM''s optimal objective value for x: ', ...
        num2str(test.objoptx)]);
end

% ADMM's optimal objective value for (x,z) that it ends up with.
if isfield(test, 'admmopt')
    disp(['ADMM''s optimal objective value for (x, z): ', ...
        num2str(test.admmopt)]);
end

% A relative error in objective terms involving ADMM.
if isfield(test, 'objerror')
    disp(['Relative error in ADMM''s objective: ', ...
        num2str(test.objerror)]);
end

% The average error in the constraint condition below, per component.
if isfield(test, 'constrainterror')
    disp(['Average error in components of constraint D*x_opt = s: ', ...
        num2str(test.constrainterror)]);
end

% A residual of the constraint below.
if isfield(test, 'constraintresidual')
    disp(['Residual ||D*x_opt - s||: ', num2str(test.constraintresidual)]);
end

% The average error between ADMM's compute x and a true x.
if isfield(test, 'xerror')
    disp(['Average error in components of x_admm: ', ...
        num2str(test.xerror)]);
end

% The residual between the optimal solution x and the one provided by ADMM.
if isfield(test, 'xresidual')
    disp(['Residual ||x* - x_admm||: ', num2str(test.xresidual)]);
end

% The number of steps performed in the execution.
if isfield(results, 'steps')
    disp(['Number of iteration steps performed: ', ...
        num2str(results.steps)]);
end

% The runtime of the call to ADMM, specifically..
if isfield(results, 'runtime')
    disp(['Runtime of ADMM call: ', num2str(results.runtime), ...
        ' seconds.']);
end

% The runtime of the entire solver itself.
if isfield(results, 'solverruntime')
    disp(['Overall runtime of solver: ', ...
        num2str(results.solverruntime), ' seconds.']);
end

% Display whether a test was successful or not and output.
if (isfield(test, 'failed') && test.failed)
    disp('TEST UNSUCCESSFUL!');
elseif (isfield(test, 'failed') && ~test.failed)
    disp('Test successful!');
end

% Output an associated reason for failing or not failing.
if isfield(test, 'failreason')
    disp(['   ', test.failreason]);
end

% Number of steps.
N = results.steps;

% Creating a line on the graph that has the true objective value, or just
% the tests objective value.
if isfield(test, 'trueobjopt')
    optline = zeros(N, 1) + test.trueobjopt;
elseif isfield(test, 'testobj')
    optline = zeros(N, 1) + test.testobj;
else
    optline = 0;
end

% Output the signal information as a plot.
if (isfield(test, 'D') && isfield(test, 's')) && isfield(test, 'testx')
    figure;
    plot(1:length(test.testx), test.testx, 'ko', 1:length(test.testx), ...
        results.xopt, 'r', 'LineWidth', 2);
    xlabel('Signal component i');
    ylabel('Signal value at i');
    legend('Noisy signal x', 'ADMM''s denoised x_{opt}');
    title('Denoising Results For Noisy Signal');
% elseif (isfield(test, 'D') && isfield(test, 's')) && isfield(test, 'truex')
%     figure;
%     plot(1:length(test.truex), test.truex, 'ko', 1:length(test.truex), ...
%         results.xopt, 'r', 'LineWidth', 2);
%     xlabel('Signal component i');
%     ylabel('Signal value at i');
%     legend('Original signal x', 'ADMM''s x_{opt}');
%     title('Results for Recovering Signal x');
end

% Decide how many subplots could be on the giant plot.
if (isfield(results.options, 'algorithm') && ...
    isfield(results.options, 'fasttype') && ... 
    strcmp(results.options.algorithm, 'fast') && ...
    strcmp(results.options.fasttype, 'weak'))
    
    plots = 3;
else
    plots = 4;
end

% If objective evals were not performed, then that's one less thing to
% graph.
if ~isfield(results, 'objevals')
    plots = plots - 1;
end

% If H-norm squared values were not evaluated we don't need to make a
% graphs of it.
if ~isfield(results, 'Hnormsq')
    plots = plots - 1;
end

% If the primal norm residuals were not evaluated, then we don't need to
% make graphs of it.
if ~isfield(results, 'pnorm')
    plots = plots - 1;
end

% If the dual norm residuals were not evaluated, then we don't need to make
% graphs of it.
if ~isfield(results, 'dnorm')
    plots = plots - 1;
end

% Check if there are any graphs to make at all, and create a figure if yes.
if (plots > 0)
    figure;
end

% Number of plots made so far.
plotssofar = 0;

% Figure for the objective value at each iteration.
if (plots > 0 && isfield(results, 'objevals'))
    % Subplot if there will be many plots.
    if plots ~= 1
        plotssofar = plotssofar + 1;
        subplot(plots, 1, plotssofar);
    end
    
    % The actual plot.
    if optline ~= 0
        plot(1:N, optline, 'b--', 1:N, results.objevals, '-k', ...
            'LineWidth', 2);
        
        % Cases on whether the provided test objective value was optitmal
        % or not.
        if isfield(test, 'trueobjopt')
            legend('True optimal objective value', ...
                'ADMM''s objective value');
        elseif isfield(test, 'testobj')
            legend('Test''s original objective value', ...
                'ADMM''s objective value');
        end
    else
        plot(results.objevals, '-k', 'LineWidth', 2);
        legend('ADMM''s objective value');
    end
    
    % Title depending on solver.
    if strcmp(solver, 'linearsvm')
        title(['Plot of objective value for each iteration for ', ...
            options.lossfunction, ' loss function']);
    else
        title('Plot of objective value for each iteration');
    end
    
    if plots == 1
        xlabel('Iteration k');
    end
    
    ylabel('Objective'); 
end


% Figure of H-norm-squared residuals. Encodes ADMM's results for x, z, and
% u as column vector w = [x z u]^T and uses a special matrix norm defined
% by matrix H.
if (plots > 0 && isfield(results, 'Hnormsq'))
    % Subplot if there will be many plots.
    if plots ~= 1
        plotssofar = plotssofar + 1;
        subplot(plots, 1, plotssofar);
    end
    
    % The actual plot.
    semilogy(1:length(results.Hnormsq), max(1e-8, results.Hnormsq), ... 
        'k', 1:N, zeros(N, 1) + results.Hnormtol, 'b--',  'LineWidth', 2);
    
    % Include a loss function in the header if given.
    if (isfield(options, 'tester') && strcmp(options.tester, 'linearsvm'))
        title(['Plot of H-Norm Squared Residuals', ...
            ' (w = [x^T z^T u^T]^T) for ', options.lossfunction, ...
            ' loss function']);
    else
        title('Plot of H-Norm Squared Residuals (w = [x^T z^T u^T]^T)');
    end
    
    % Decide whether to output the iteration or if there will be another
    % subplot after this.
    if plots == 1 || plotssofar == plots
        xlabel('Iteration k');
    end
    
    % Pretty stuff for the plot.
    legend('H-Norm', 'Threshold');
    ylabel('||w^{k - 1} - w^k||_H^2');
end


% In the case of pnorms, plot their values.
if (plots > 0 && isfield(results, 'pnorm'))
    % Subplot if there will be many plots.
    if plots ~= 1
        plotssofar = plotssofar + 1;
        subplot(plots, 1, plotssofar);
    end
    
    % The actual plot.
    semilogy(1:N, max(1e-8, results.pnorm), 'k', ...
        1:N, results.perr, 'b--',  'LineWidth', 2);
    
    % Again, handle the issue of stating the loss function being used.
    if (isfield(options, 'tester') && strcmp(options.tester, 'linearsvm'))
        title(['Plot of Primal Residual Norm for ', ...
            options.lossfunction, ' loss function']);
    else
        title('Plot of Primal Residual Norm');
    end
    
    % Pretty things on the graphs.
    legend('Primal Norm', 'Primal Error');
    ylabel('||Ax^k - Bz^k - c||_2');
    
    % Check if its the last plot to make and if so output the the y label
   %  iterations.
    if plots == 1 || plotssofar == plots
        xlabel('Iteration k');
    end
end


% Decide whether you need to plot the dual residual norms.
if (plots > 0 && isfield(results, 'dnorm'))
    % Subplot if there will be many plots.
    if plots ~= 1
        plotssofar = plotssofar + 1;
        subplot(plots, 1, plotssofar);
    end
    
    % The actual plot.
    semilogy(1:N, max(1e-8, results.dnorm), 'k', ...
        1:N, results.derr, 'b--', 'LineWidth', 2);
    
    % Loss function or not in the title.
    if (isfield(options, 'tester') && strcmp(options.tester, 'linearsvm'))
        title(['Plot of Dual Residual Norm for ', ...
            options.lossfunction, ' loss function']);
    else
        title('Plot of Dual Residual Norm');
    end
    
    % Pretty stuff for the graph.
    legend('Dual Norm', 'Dual Error');
    ylabel('||\rho*A^T*B*(z^k-z^{k-1})||_2');
    
    % If last plot, put the iteration count on the xlabel.
    if plots == 1 || plotssofar == plots
        xlabel('Iteration k');
    end
end


% Check if you need to plot the Accelerated ADMM's dvals.
if (plots > 0 && isfield(results, 'dvals'))
    % Subplot if there will be many plots.
    if plots ~= 1
        plotssofar = plotssofar + 1;
        subplot(plots, 1, plotssofar);
    end
    
    % Figure of d values
    semilogy(1:N, max(1e-8, results.dvals), 'k', ...
        1:N, zeros(N, 1) + results.dvaltol, 'b--', 'LineWidth', 2);
    
    % Output the loss function or not in the title.
    if (isfield(options, 'tester') && strcmp(options.tester, 'linearsvm'))
        title(['Plot of Accelerated ADMM''s Residual Norms for ', ...
            options.lossfunction, ' loss function']);
    else
        title('Plot of Accelerated ADMM''s Residual Norms');
    end
    
    % Pretty stuff for the plot.
    legend('Accelerated Residual Norm', 'Convergence point');
    ylabel('1/\rho||u^k - u_{hat}^k||^2 + \rho||B(z^k - z_{hat}^k)||^2');
    xlabel('Iteration k');
end
    
end
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------