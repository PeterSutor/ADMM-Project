% Coded by:     Peter Sutor Jr.
% Last edit:    3/6/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Tests the simple model problem solver using ADMM. The model problem to
% solve is minimizing for x the objective function:
%    obj(x) = 1/2*(||P*x - r||_2)^2 + 1/2*(||Q*x - s||_2)^2,
% where P is a m by n matrix, Q is an m by n matrix, and r and s are
% column vectors of length m. Thus, x is a column vector of length n.
% See function model for more details on how to solve the model problem
% using ADMM.
% 
% The test creates random data for P, Q, r, and s, of size rows by cols,
% given as inputs. The seed specifies the seed value to use in random
% number generation, for repeatability. The value errtol is the relative
% error tolerance from the true solution that is allowed for the test to be
% successful. The variable quiet determines whether to suppress output and
% graphs.
%
% Note that we can always solve this model problem exactly using calculus:
%    min_x(obj(x)) <--> x such that grad_x(obj(x)) := 0
%                  <--> P^T*(P*x - r) + Q^T*(Q*x - s) = 0
%                  <--> (P^T*P + Q^T*Q)*x = P^T*r + Q^T*s
%                  <--> x = (P^T*P + Q^T*Q)^-1*(P^T*r + Q^T*s)
% We check the ADMM solution against this value for correctness.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If modeltest is executed with no arguments, it will create a demo
% test of random data of size m = n = 2^7. The random values are generated
% by the function rng with the 'shuffle' option to be seedless. The rows
% and cols are set to 2^7, with an errtol of 0.001, and quiet = 0. The
% options struct is set to empty, thus using ADMM's default setup.

function [results, test] = ...
    modeltest(seed, rows, cols, errtol, quiet, options)
% INPUTS ------------------------------------------------------------------
% seed:     The seed used to generate random data in the test.
% rows:     The variable m in the m by n problem specified in the model
%           being tested, described in the description above.
% cols:     The variable n in the m by n problem specified in the model
%           being tested, described in the description above.
% errtol:   The relative error from the true solution that is allowed for
%           the test to be considered successfully convergent. Typical
%           values are 0.01 or 0.001, corresponding to 1% or 0.1% accuracy
%           to the true solution.
% quiet:    A binary value specifying whether to show output and graphs for
%           the test.
% options:  A struct containing options customizing the ADMM execution. If
%           no options are provided (an empty or irrelevant struct is given
%           as the options parameter), default settings are applied, as 
%           mentioned in the user manual.
% 
% OUTPUTS -----------------------------------------------------------------
% results:  A struct containing the results of the execution, including the
%           optimized values for x, z and u that optimize the objective for
%           x, runtime evaluations, records of each iteration, etc. Consult
%           the user manual for more details on the results struct, or
%           simply check what the variable contains in the Matlab 
%           interpreter after execution.
% test:     A struct containing the details of the random test that was
%           performed, including the random data generated, how correct it
%           was, whether it reached the correct tolerance (successful test)
%           and so on. Consult the user manual for more details on the test
%           struct, or simply check what the variable contains in the 
%           Matlab interpreter after execution.
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

% Check if no arguments were given. If so, run the default test, otherwise
% proceed with user's inputs.
if nargin == 0
    % Demo test's default settings.
    rng('shuffle');
    rows = 2^7;
    cols = 2^7;
    errtol = 0.001;
    quiet = 0;
    options = struct();
else
    % Error checking on input.
    [seed, rows, cols, errtol, quiet] = ...
        errorcheck(seed, rows, cols, errtol, quiet, options);

    rng('default');                 % Default RNG.
    rng(seed);                      % Seed the RNG.
end

% Generate normally distributed, random data for the model test.
P = randn(rows, cols);
Q = randn(rows, cols);
r = randn(rows, 1);
s = randn(rows, 1);

% This is the exact true solution and corresponding true objective value
% for the random data generated.
truexopt = (P'*P + Q'*Q) \ (P'*r + Q'*s);
trueobjopt = 1/2*norm(P*truexopt - r, 'fro')^2 + ...
    1/2*norm(Q*truexopt - s, 'fro')^2;

% Specify options in test outside of those given in options struct.
options.objevals = 1;               % We want objective evaluations.
options.maxiters = 10000;           % Run up to 10000 iterations.
options.convtest = 1;               % We want to test that iterations are 
                                    % converging.
options.solver = 'model';           
options.stopcond = 'both';          % Use both standard stopping conditions
                                    % and H-norms-squared.

% Run the model solver on the random data.
results = model(P, Q, r, s, options);

% Get some results from our model solver run.
xopt = results.xopt;                % The optimal solution model returned.
admmopt = results.objopt;           % The corresponding ADMM objective
                                    % value estimated for x and z.
                                    
% The objective value computed by the model solver.
objopt = 1/2*norm(P*xopt - r, 'fro')^2 + 1/2*norm(Q*xopt - s, 'fro')^2;

% Check if true objective is non-zero so relative error can be computed.
if abs(trueobjopt) > 0
    % Compute the relative error between true solution and model solver's.
    objerror = abs(1 - objopt/trueobjopt);
    
    % Compute x residual.
    xresidual = norm(truexopt - xopt, 'fro');
    
    % Check if we are within the error tolerance errtol to determine if
    % test succeeded or failed.
    if (objerror <= errtol && xresidual <= errtol)
        failed = 0;
        failreason = ['ADMM''s objective and x values match the true', ...
            ' solution to specified relative error tolerance ', ...
            num2str(errtol)];
    elseif (objerror <= errtol)
        failed = 1;
        failreason = ['ADMM''s objective is within relative error', ...
            ' tolerance ', num2str(errtol), ...
            ', but the x solution is not within this tolerance of the', ...
            ' true solution'];
    elseif (xresidual <= errtol)
        failed = 1;
        failreason = ['ADMM''s x solution is within relative error', ...
            ' tolerance ', num2str(errtol), ' of the true x, but', ...
            ' the objective value is not'];
    else
        failed = 1;
        failreason = ['Both ADMM''s objective value and solution x', ...
            ' are not within a relative error tolerance ', ...
            num2str(errtol), ' of the true solution.'];
    end
else
    % Mention that we could not compute error, if output allowed.
    if ~quiet
        disp('True objective too close to 0 to compute error!');
    end
    
    % Set the relative error as not a number, and that the test failed,
    % vacuously.
    objerror = NaN;
    failed = 0;
end

% Set the output of the test struct that will be returned.
test.P = P;                                 % The random P used in test.
test.Q = Q;                                 % The random Q used in test.
test.r = r;                                 % The random r used in test.
test.s = s;                                 % The random s used in test.
test.truexopt = truexopt;                   % The true solution.
test.trueobjopt = trueobjopt;               % True x's objective value.
test.xopt = xopt;                           % Model solver's solution.
test.admmopt = admmopt;                     % ADMM's objective value.
test.failed = failed;                       % If we passed or failed test.
test.objerror = objerror;                   % Computed relative error.
test.steps = results.steps;                 % Number of steps to converge.
test.errtol = errtol;                       % The error tolerance used.
test.failreason = failreason;               % Pass / fail message.

% Residual between true x solution and model solver's.
test.xresidual = xresidual;

% If output not suppressed, show output and graphs of convergence.
if ~quiet
    showresults(results, test, options);  
end

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Simply checks for any user error on same input, stops if error is
% detected. If input is correct, return info needed for problem.

function [seed, rows, cols, errtol, quiet] = ...
    errorcheck(seed, rows, cols, errtol, quiet, options)
% INPUTS ------------------------------------------------------------------
% Same as for the modeltest function.
% 
% OUTPUTS -----------------------------------------------------------------
% Same as the inputs, just potentially altered values for strange input.
% -------------------------------------------------------------------------


% Check for invalid input and report errors / terminate if detected.
if (~isnumeric(seed) || floor(seed) < 0)
    error('Given RNG seed is not a nonnegative integer!');
elseif (~isnumeric(rows) || floor(real(rows)) < 0)
    error('Given number of rows is not a nonnegative integer!');
elseif (~isnumeric(cols) || floor(real(cols)) < 0)
    error('Given number of columns is not a nonnegative integer!');
elseif (~isnumeric(errtol) || real(errtol) <= 0)
    error('Given error tolerance errtol is not a positive number!');
else
    seed = floor(real(seed));   % Set the seed to valid value.
    rows = floor(real(rows));   % Set the rows to valid value.
    cols = floor(real(cols));   % Set the columnss to valid value.
    errtol = real(errtol);      % Se the error tolerance to valid value.
end

% Only set output to be suppressed if specified.
if (~isnumeric(quiet) || floor(real(quiet)) ~= 1)
    quiet = 0;
else
    quiet = 1;
end

% Check that the given options is actually a struct...
if ~isstruct(options)
    error(['Given options is not a struct! Please check your', ...
        ' arguments and try again.']);
end

end
% -------------------------------------------------------------------------