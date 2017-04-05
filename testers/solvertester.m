% Coded by:     Peter Sutor Jr.
% Last edit:    4/12/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% This function tests a solver in ADMM multiple times, over increasing
% scales, and checks that each test statisfies a certain tolerance for
% error from the true solution. The tester tests random data on order of
% some function of integer scale, with increasing values increasing the
% scale. Each programmed solver and tester for it has a default function
% for scale. These functions have some settings to play with, or one can
% provide their own scale function for the test. The tester will test a
% given amount of random trials for each scale from some provided minimum
% value to a maximum. If desired, the tester will show some output about
% the testing session, such as a plot of average runtime over each scale.
% The output is a struct called results that contains the results for EACH
% trial.
%
% Consult the manual to see what results will contain. Alternatively, one
% can check what it contains after execution by just typing its name in the
% interpreter.
%
% NOTE: If function solvertester is executed with no inputs, it will run a
% demo testing session on the model problem solver from minimum scale 2 to
% 8, with 10 trials per scale, random square problems, using the standard
% scaler function of problems size 2^scale, and error tolerance at the
% default 0.001, showing the results.

function [results] = ...
    solvertester(solver, minscale, maxscale, trials, showplots, options)
% INPUTS ------------------------------------------------------------------
% solver:       A character array containing the name of the solver to
%               test. For example, 'basispursuit' or 'linearprogram'.
% minscale:     The minimum scale to use as a parameter for the scaling
%               function for the solver. Should be a positive integer.
% maxscale:     The maximum scale to use as a parameter for the scaling
%               function for the solver. Should be a positive integer
%               greater than or equal to minscale.
% trials:       The number of random test trials to perform for each scale.
%               Should be a positive integer.
% showplots:    A binary value indicating whether or not you want plots of
%               the results to be shown. Any input other than 1 is treated
%               as a 0 (don't show plots).
% options:      The options struct to pass to ADMM / solver for the trials,
%               for customizing the test. Also, this struct can include
%               options for solvertester. For example, this includes the
%               testtype ('fat', 'skinny', 'square' matrices), an optional
%               scaler function to use (a function handle options.scaler),
%               the error tolerance to use (options.error), or a seed for
%               the random number generators in the tests (options.seed).
%
% OUTPUTS -----------------------------------------------------------------
% results:      Contains results for EACH trial. This includes: a matrix of
%               runtimes, where each row is a different scale and each
%               column is a different trial; a matrix of failures, where
%               each row is a different scale an each column is a different
%               trial, with a binary value indicating failure to meet the
%               error tolerance or not; a vector of average runtimes, where
%               each row is a different scale and contains the average
%               runtime over those trials; a single binary failure value
%               denoting whether each test was successful or not; and
%               matrices of structs, where one represents the struct
%               decribing the test parameters / results, and one that
%               represents the results struct ADMM returns for that test.
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
    solver = 'model';
    minscale = 2;
    maxscale = 8;
    trials = 10;
    showplots = 1;
    options = struct();
else
    % Error checking on input.
    [solver, minscale, maxscale, trials, showplots] = ...
        errorcheck(solver, minscale, maxscale, trials, showplots, options);
end


% Set RNG to shuffle, as default, if seed not provided.
if ~isfield(options, 'seed')
    rng('shuffle');
    rng(rand);
end

% Populate the error tolerance field with default values.
if ~isfield(options, 'errtol')
    if strcmp(solver, 'basispursuit') || strcmp(solver, 'linearprogram')
        options.errtol = 1e-10;
    elseif strcmp(solver, 'linearsvm')
        options.errtol = 0.05;
    else
        options.errtol = 1e-3;
    end
end

% Set the type of test to default if it isn't populated.
if ~isfield(options, 'testtype')
    options.testtype = 'default';
end

% Set the scaler to default if it isn't populated.
if (~isfield(options, 'scaler') || ~isa(options.scaler, 'function_handle'))
    options.scaler = 'default';
end

% The number of scales we are testing.
scales = maxscale - minscale + 1;

if strcmp(solver, 'linearsvm')
    % Initialize data storing vectors / matrices.
    runtimes = zeros(scales, trials, 2);% Matrix of runtimes.
    failed = zeros(scales, trials, 2);  % Matrix of failure flags.
    avetimes = zeros(scales, 1, 2);     % Vector of average times per scale
else
    % Initialize data storing vectors / matrices.
    runtimes = zeros(scales, trials);   % Matrix of runtimes.
    failed = zeros(scales, trials);     % Matrix of failure flags.
    avetimes = zeros(scales, 1);        % Vector of average times per scale
end

% Purpose: Iterate over scales and perform tests for each scale.
for i = minscale:maxscale 
    r = i - minscale + 1;           % The current scale index (row index).
    
    % Purpose: Iterate over each trial to perform and perform it.
    for c = 1:trials
        % Create a random seed for the test's RNG.
        options.seed = floor(rand*intmax);
        
        % Purpose: Select the correct solver to apply and run its tester
        % function.
        switch(solver)
            % Model problem tester.
            case 'model'
                [runtimes(r, c), failed(r, c), testtrial, ...
                    testresult] = modeltester(i, options);
            % Basis Pursuit tester.
            case 'basispursuit'
                [runtimes(r, c), failed(r, c), testtrial, ...
                    testresult] = basispursuittester(i, options);
            % Linear Program tester.
            case 'linearprogram'
                [runtimes(r, c), failed(r, c), testtrial, ...
                    testresult] = linearprogramtester(i, options);
            % LASSO problem tester.
            case 'lasso'
                [runtimes(r, c), failed(r, c), testtrial, ...
                    testresult] = lassotester(i, options);
            % Total Variation Minimization tester.
            case 'totalvariation'
                [runtimes(r, c), failed(r, c), testtrial, ...
                    testresult] = totalvariationtester(i, options);
            % Linear SVM tester.
            case 'linearsvm'
                [runtime, fail, testtrial, ...
                    testresult] = linearsvmtester(i, options);
                runtimes(r, c, 1) = runtime(1);
                runtimes(r, c, 2) = runtime(2);
                failed(r, c, 1) = fail(1);
                failed(r, c, 2) = fail(2);
            % Least Absolute Deviations.
            case 'lad'
                [runtimes(r, c), failed(r, c), testtrial, ...
                    testresult] = ladtester(i, options); 
            % Huber Fitting
            case 'huberfit'
                [runtimes(r, c), failed(r, c), testtrial, ...
                    testresult] = huberfittester(i, options); 
            % Covariance Selection.
            case 'covarianceselection'
                [runtimes(r, c), failed(r, c), testtrial, ...
                    testresult] = covarianceselectiontester(i, options); 
            otherwise
                error(['Given string argument solver does not match', ...
                    ' an existing type of solver to batch test on.']);
        end
        
        % Record the test trial and its results in struct matrix.
        testtrials(r, c) = testtrial; %#ok<AGROW>
        testresults(r, c) = testresult; %#ok<AGROW>
    end
    
    % Account for the Linear SVM solver returning two separate tests.
    if strcmp(solver, 'linearsvm')
        % Compute the average runtime for this scale.
        avetimes(r, 1, 1) = sum(runtimes(r, :, 1)) / trials;
        avetimes(r, 1, 2) = sum(runtimes(r, :, 2)) / trials;
    else
        % Compute the average runtime for this scale.
        avetimes(r) = sum(runtimes(r, :)) / trials;
    end
    
end

% Record all results from testing in the return results struct.
results.runtimes = runtimes;        % Matrix of runtimes.
results.failed = failed;            % Matrix of failure flags.
results.avetimes = avetimes;        % Vector of average runtime per scale.
results.testtrials = testtrials;    % Test trial data / results.
results.testresults = testresults;  % ADMM's returned results struct.

% Check if any failures occurred and report accordingly - set the flag for
% global failure in results struct.
if (sum(sum(sum(failed))) > 0)
    results.failure = 1;
    disp('SOLVERTESTER: Failure detected!!!');
else
    results.failure = 0;
    disp(['SOLVERTESTER for ', upper(solver), ...
        ': All trials and scales successfully finished.']);
end

% Create plots of results.
if showplots && strcmp(solver, 'linearsvm')
    figure;
    
    % Plot of average runtime per scale.
    %subplot(1, 3, 1);
    semilogy(minscale:maxscale, avetimes(:, 1, 1), minscale:maxscale, ...
        avetimes(:, 1, 2));
    title('Plot of average ADMM run-time over successive scales.');
    xlabel('Scale used');
    ylabel('Average run-time in seconds');
    legend('Hinge loss', '0-1 loss');
elseif showplots
    figure;
    
    % Plot of average runtime per scale.
    %subplot(1, 3, 1);
    semilogy(minscale:maxscale, avetimes);
    title('Plot of average ADMM run-time over successive scales.');
    xlabel('Scale used');
    ylabel('Average run-time in seconds');
end

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Simply checks for any user error on same input, stops if error is
% detected. If input is correct, return info needed for problem.

function [solver, minscale, maxscale, trials, showplots] = ...
    errorcheck(solver, minscale, maxscale, trials, showplots, options)
% INPUTS ------------------------------------------------------------------
% Same as for the solvertester function.
% 
% OUTPUTS -----------------------------------------------------------------
% Same as the inputs, just potentially altered values for strange input.
% -------------------------------------------------------------------------


% Check that the inputs are of the datatype they need to be and account for
% any strangeness in input.
if ~ischar(solver)
    error('Given argument solver is not a character array!');
elseif (~isnumeric(minscale) || floor(minscale) <= 0)
    error('Given argument minscale is not a positive integer!');
elseif (~isnumeric(maxscale) || floor(maxscale) <= 0)
    error('Given argument maxscale is not a positive integer!');
elseif (minscale > maxscale)
    error('The minscale argument is bigger than maxscale!');
elseif (~isnumeric(trials) || floor(trials) <= 0)
    error('Given argument trials is not a positive integer!');
else
    solver = lower(solver);             % Lowercase the solver name.
    minscale = real(floor(minscale));   % Obtain integer version.
    maxscale = real(floor(maxscale));   % Obtain integer version.
    trials = real(floor(trials));       % Obtain integer version.
end

% Check whether to show plots or not, and set showplots appropriately.
if (isnumeric(showplots) && showplots ~= 1)
    showplots = 0;
end

% Check that the given options is actually a struct...
if ~isstruct(options)
    error('Given options is not a struct! At least pass empty struct!');
end

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Helper function for running a model test trial. Sets up input for running
% function modeltest and then runs it on this input to perform a test
% trial.

function [runtime, failed, testtrial, testresults] = ...
    modeltester(scale, options)
% INPUTS ------------------------------------------------------------------
% scale:        The scale parameter to use for the scaler function.
%               Provided in options.scaler as a function handle or
%               specified to differ from default via options.testtype =
%               'fat' or 'skinny'. The 'fat' type is a 2^(scale - 1) by
%               2^scale matrix problem (a fat matrix) and 'skinny' the
%               reverse (a skinny matrix). Default (unspecified or
%               different value) is a square 2^scale by 2^scale matrix
%               problem.
% options:      Same as options in solvertester function above. This one
%               differs only by including a randomly generated seed.
% 
% OUTPUTS -----------------------------------------------------------------
% runtime:      The runtime reported by ADMM.
% failed:       Flag for if trial failed (didn't meet specified tolerance).
% testtrial:    Test struct returned by solver with test info and results.
% testresults:  Results struct returned by solver with ADMM execution info.
% -------------------------------------------------------------------------


% Decide what to use for scaler under given scale parameter.
if isa(options.scaler, 'function_handle')   % User-defined scalar function.
    sizes = options.scaler(scale);
    m = sizes(1);
    n = sizes(2);
elseif strcmp(options.testtype, 'fat')      % Fat matrix (double columns).
    m = 2^(scale - 1);
    n = 2^scale;
elseif strcmp(options.testtype, 'skinny')   % Skinny matrix (double rows).
    m = 2^scale;
    n = 2^(scale - 1);
else                                        % Default square matrix.
    m = 2^scale;
    n = m;
end

% Run the tester for this problem.
[testresults, testtrial] = ...
    modeltest(options.seed, m, n, options.errtol, 1, options);

% Record runtime and whether test failed or not.
runtime = testresults.solverruntime;
failed = testtrial.failed;

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Helper function for running a Basis Pursuit test trial. Sets up input for
% running function basispursuittest and then runs it on this input to
% perform a test trial.

function [runtime, failed, testtrial, testresults] = ...
    basispursuittester(scale, options)
% INPUTS ------------------------------------------------------------------
% scale:        The scale parameter to use for the scaler function.
%               Provided in options.scaler as a function handle or only
%               default one will run. The default scale function is the
%               ceiling of (2^scale)/5 by 2^scale problem, i.e., an
%               underdetermined system / fat matrix problem..
% options:      Same as options in solvertester function above. This one
%               differs only by including a randomly generated seed.
% 
% OUTPUTS -----------------------------------------------------------------
% Same outputs as in function modeltester, above.
% -------------------------------------------------------------------------


% Decide what to use for scaler under given scale parameter.
if isa(options.scaler, 'function_handle')   % User defined scalar function.
    sizes = options.scaler(scale);
    m = sizes(1);
    n = sizes(2);
else                                        % Default scaler function.
    n = 2^scale;
    m = ceil(n/5);
end

% Run the test trial on the chosen solver.
[testresults, testtrial] = ...
    basispursuittest(options.seed, m, n, options.errtol, 1, options);

% Record the runtime and failed flag reported by tester.
runtime = testresults.solverruntime;
failed = testtrial.failed;

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Helper function for running a Total Variation Minimization test trial. 
% Sets up input for running function totalvariationtest and then runs it on
% this input to perform a test trial.

function [runtime, failed, testtrial, testresults] = ...
    totalvariationtester(scale, options)
% INPUTS ------------------------------------------------------------------
% Same inputs as function modeltester, above.
% 
% OUTPUTS -----------------------------------------------------------------
% Same outputs as in function modeltester, above.
% -------------------------------------------------------------------------


% Decide what to use for scaler under given scale parameter.
if isa(options.scaler, 'function_handle')   % User-defined scalar function.
    sizes = options.scaler(scale);
    n = sizes(1);
else                                        % Default square matrix.
    n = 2^scale;
end

% Run the test trial on the chosen solver.
[testresults, testtrial] = ...
    totalvariationtest(options.seed, n, options.errtol, 1, options);

% Record runtime and failed flag for the test trial.
runtime = testresults.solverruntime;
failed = testtrial.failed;

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Helper function for running a Linear Program test trial. Sets up input
% for running function linearprogramtest and then runs it on this input to
% perform a test trial.

function [runtime, failed, testtrial, testresults] = ...
    linearprogramtester(scale, options)
% INPUTS ------------------------------------------------------------------
% Same inputs as function modeltester, above.
% 
% OUTPUTS -----------------------------------------------------------------
% Same outputs as in function modeltester, above.
% -------------------------------------------------------------------------


% Decide what to use for scaler under given scale parameter.
if isa(options.scaler, 'function_handle')   % User-defined scalar function.
    sizes = options.scaler(scale);
    m = sizes(1);
    n = sizes(2);
elseif strcmp(options.testtype, 'fat')      % Fat matrix (double columns).
    m = 2^(scale - 1);
    n = 2^scale;
elseif strcmp(options.testtype, 'skinny')   % Skinny matrix (double rows).
    m = 2^scale;
    n = 2^(scale - 1);
else                                        % Default square matrix.
    m = 2^scale;
    n = m;
end

% Run the test trial on the chosen solver.
[testresults, testtrial] = ...
    linearprogramtest(options.seed, m, n, options.errtol, 1, options);

% Record runtime and failed flag for the test trial.
runtime = testresults.solverruntime;
failed = testtrial.failed;

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Helper function for running a Linear SVM test trial. Sets up input for
% running function linearvmtest and then runs it on this input to perform a
% test trial.

function [runtime, failed, testtrial, testresults] = ...
    linearsvmtester(scale, options)
% INPUTS ------------------------------------------------------------------
% Same inputs as function modeltester, above.
% 
% OUTPUTS -----------------------------------------------------------------
% Same outputs as in function modeltester, above.
% -------------------------------------------------------------------------


% Decide what to use for scaler under given scale parameter.
if isa(options.scaler, 'function_handle')   % User-defined scalar function.
    sizes = options.scaler(scale);
    mpos = sizes(1);
    mneg = sizes(2);
elseif strcmp(options.testtype, 'morepos')  % More negative labelled rows.
    mpos = ceil(2^(scale - 1)/2);
    mneg = 2^scale;
elseif strcmp(options.testtype, 'moreneg')  % More positive labelled rows.
    mpos = 2^scale;
    mneg = ceil(2^(scale - 1)/2);
else                                        % Default equal sized classes.
    mpos = 2^scale;
    mneg = mpos;
end

% Set a default separation parameter.
if ~isfield(options, 'sep')
    options.sep = 0.2;
end

% Set which loss function to test on.
if ~isfield(options, 'lossfunction')
    options.lossfunction = 'hinge';
end

% Run the test trial on the chosen solver.
[testresults, testtrial] = linearsvmtest(options.seed, mpos, mneg, ...
    options.sep, options.errtol, 1, options);

% Record runtime and failed flag for the test trial.
runtime = [testresults.hingeloss.solverruntime; ...
    testresults.zoloss.solverruntime];
failed = [testtrial.hingeloss.failed; testtrial.zoloss.failed];

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Helper function for running a LASSO test trial. Sets up input for running
% function lassotest and then runs it on this input to perform a test
% trial.

function [runtime, failed, testtrial, testresults] = ...
    lassotester(scale, options)
% INPUTS ------------------------------------------------------------------
% Same inputs as function modeltester, above.
% 
% OUTPUTS -----------------------------------------------------------------
% Same outputs as in function modeltester, above.
% -------------------------------------------------------------------------


% Decide what to use for scaler under given scale parameter.
if isa(options.scaler, 'function_handle')   % User-defined scalar function.
    sizes = options.scaler(scale);
    m = sizes(1);
    n = sizes(2);
elseif strcmp(options.testtype, 'fat')      % Fat matrix (double columns).
    m = 2^(scale - 3);
    n = 2^scale;
elseif strcmp(options.testtype, 'square')   % Square matrix.
    m = 2^scale;
    n = m;
else                                        % Skinny matrix (8x more rows).
    m = 2^scale;
    n = 2^(scale - 3);
end

% Run the test trial on the chosen solver.
[testresults, testtrial] = ...
    lassotest(options.seed, m, n, options.errtol, 1, options);

% Record runtime and failed flag for the test trial.
runtime = testresults.solverruntime;
failed = testtrial.failed;

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Helper function for running a Least Absolute Deviations test trial. Sets
% up input for running the function ladtest and then runs it on this input
% to perform a test trial.

function [runtime, failed, testtrial, testresults] = ...
    ladtester(scale, options)
% INPUTS ------------------------------------------------------------------
% Same inputs as function modeltester, above.
% 
% OUTPUTS -----------------------------------------------------------------
% Same outputs as in function modeltester, above.
% -------------------------------------------------------------------------


% Decide what to use for scaler under given scale parameter.
if isa(options.scaler, 'function_handle')   % User-defined scalar function.
    sizes = options.scaler(scale);
    m = sizes(1);
    n = sizes(2);
elseif strcmp(options.testtype, 'fat')      % Fat matrix (double columns).
    m = 2^(scale - 3);
    n = 2^scale;
elseif strcmp(options.testtype, 'square')   % Square matrix.
    m = 2^scale;
    n = m;
else                                        % Skinny matrix (8x more rows).
    m = 2^scale;
    n = 2^(scale - 3);
end

% Run the test trial on the chosen solver.
[testresults, testtrial] = ...
    ladtest(options.seed, m, n, options.errtol, 1, options);

% Record runtime and failed flag for the test trial.
runtime = testresults.solverruntime;
failed = testtrial.failed;

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Helper function for running a Huber Fitting test trial. Sets up input for
% running the function huberfittest and then runs it on this input to 
% perform a test trial.

function [runtime, failed, testtrial, testresults] = ...
    huberfittester(scale, options)
% INPUTS ------------------------------------------------------------------
% Same inputs as function modeltester, above.
% 
% OUTPUTS -----------------------------------------------------------------
% Same outputs as in function modeltester, above.
% -------------------------------------------------------------------------


% Decide what to use for scaler under given scale parameter.
if isa(options.scaler, 'function_handle')   % User-defined scalar function.
    sizes = options.scaler(scale);
    m = sizes(1);
    n = sizes(2);
elseif strcmp(options.testtype, 'fat')      % Fat matrix (double columns).
    m = 2^(scale - 3);
    n = 2^scale;
elseif strcmp(options.testtype, 'square')   % Square matrix.
    m = 2^scale;
    n = m;
else                                        % Skinny matrix (8x more rows).
    m = 2^scale;
    n = 2^(scale - 3);
end

% Run the test trial on the chosen solver.
[testresults, testtrial] = ...
    huberfittest(options.seed, m, n, options.errtol, 1, options);

% Record runtime and failed flag for the test trial.
runtime = testresults.solverruntime;
failed = testtrial.failed;

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Helper function for running a Sparse Inverse Covariance Selection test
% trial. Sets up input for running the function covarianceselectiontest and
% then runs it on this input to perform a test trial.

function [runtime, failed, testtrial, testresults] = ...
    covarianceselectiontester(scale, options)
% INPUTS ------------------------------------------------------------------
% Same inputs as function modeltester, above.
% 
% OUTPUTS -----------------------------------------------------------------
% Same outputs as in function modeltester, above.
% -------------------------------------------------------------------------


% Decide what to use for scaler under given scale parameter.
if isa(options.scaler, 'function_handle')   % User-defined scalar function.
    sizes = options.scaler(scale);
    m = sizes(1);
    n = sizes(2);
elseif strcmp(options.testtype, 'fat')      % Fat matrix (double columns).
    m = 2^(scale - 3);
    n = 2^scale;
elseif strcmp(options.testtype, 'square')   % Square matrix.
    m = 2^scale;
    n = m;
else                                        % Skinny matrix (8x more rows).
    m = 2^scale;
    n = 2^(scale - 3);
end

% Run the test trial on the chosen solver.
[testresults, testtrial] = ...
    covarianceselectiontest(options.seed, m, n, ...
    options.errtol, 1, options);

% Record runtime and failed flag for the test trial.
runtime = testresults.solverruntime;
failed = testtrial.failed;

end
% -------------------------------------------------------------------------