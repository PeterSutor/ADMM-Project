% Coded by:     Peter Sutor Jr.
% Last edit:    5/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Tests the Sparse Inverse Covariance Selection problem. Sparse Inverse
% Covariance Selection minimizes for x the objective function:
%    obj(X) = trace(S*X) - log(det(X)) + lambda*||X||_1, 
% where S = covariance(D) is a square matrix of the covariance of D. Thus,
% X is a square matrix as well. The parameter lambda here is the l_1
% regularization parameter. See function covarianceselection for more
% details about how to solve this problem with ADMM.
% 
% The test creates random data for D, of size rows by cols, given as 
% inputs. The seed specifies the seed value to use in random number 
% generation, for repeatability. The variable quiet determines whether to
% suppress output and graphs.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If covarianceselectiontest is executed with no arguments, it will
% create a demo test of random data of size m = 2^8, n = 2^6. The random
% values are generated by the function rng with the 'shuffle' option to be
% seedless. The rows and cols are set to m and n, and quiet = 0. The
% options struct is set to empty, thus using ADMM's default setup.

function [results, test] = ...
    covarianceselectiontest(seed, rows, cols, errtol, quiet, options)
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
    rows = 2^9;
    cols = 2^6;
    errtol = 0.001;
    quiet = 0;
    options = struct();
else
    % Error checking on input.
    [seed, rows, cols, errtol, quiet] = ...
        errorchecker(seed, rows, cols, errtol, quiet, options);

    rng('default');                 % Default RNG.
    rng(seed);                      % Seed the RNG.
end

% Set lambda parameter, if given, otherwise set a default value.
if isfield(options, 'lambda')
    lambda = errorcheck();
else
    lambda = 1;
end

% Generate a sparse symmetric inverse covariance matrix to test with.
Sinv = diag(abs(ones(cols, 1)));
indices = randsample(cols^2, ceil(0.001*cols^2));
Sinv(indices) = ones(numel(indices), 1);
Sinv = Sinv + Sinv';             % Make sigma symmetric.

% Now make Sigma also positive definite.
if min(eig(Sinv)) < 0
    Sinv = Sinv + 1.1*abs(min(eig(Sinv)))*eye(cols);
end

S = inv(Sinv);                     % Compute the inverse of covariance.

% Generate Gaussian samples, which serve as our data set D.
D = mvnrnd(zeros(1, cols), S, rows);

% Our objective function to minimize.
obj = @(X, Z) trace(S*X) - log(det(X)) + lambda*norm(Z(:), 1); %#ok<MINV>

% Our true optimal objective function value.
trueobjopt = obj(Sinv, Sinv);

% Set the options struct for ADMM.
options.objevals = 1;
options.maxiters = 1000;
options.convtest = 1;
options.tester = 'covarianceselection';

% Solve problem with Sparse Inverse Covariance Selection solver.
results = covarianceselection(D, lambda, options);

% Obtain relevant data from ADMM solution.
xopt = results.xopt;                % ADMM's optimal x solution.
admmopt = results.objopt;           % ADMM's optimal objective value.
objopt = obj(xopt, xopt);           % The actual objective value computed.

% Check to see if the test was successful or not - the objective value
% should be smaller.
if (objopt < trueobjopt)
    failed = 0;
else
    failed = 1;
end

% Populate the test struct with details about the test and its results.
test.truexopt = Sinv;
test.trueobjopt = trueobjopt;
test.xopt = xopt;
test.admmopt = admmopt;
test.objopt = objopt;
test.failed = failed;
test.steps = results.steps;
test.errtol = errtol;
test.tester = 'covarianceselection';

% Show results if the user wanted to see them. 
if ~quiet
    % Run show results on output.
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
    errorchecker(seed, rows, cols, errtol, quiet, options)
% INPUTS ------------------------------------------------------------------
% Same as for the covarianceselectiontest function.
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