% Coded by:     Peter Sutor Jr.
% Last edit:    4/12/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Tests the Linear SVM solver using ADMM. Trains Linear Support Vectors on
% a given data matrix D, returning dividing hyperplane as optimal x
% parameter. ADMM minimizes for x the objective function:
%    obj(x) = 1/2*||x||_2^2 + C*hinge(D*x)
% where hinge(v) = sum_i{max(1 - ell_i*v_i, 0)} is the hinge loss function,
% D is vector/matrix of data (each row is a data point/vector) on which we
% train the SVM, and C is a regularization parameter that specifies whether
% we use hard linear classification (C = 0) or soft (and to what degree)
% classification (C > 0), where we allow some points to pass the dividing
% hyperplane from either class. The vector ell is a vector of 
% classification labels for our classifier; we use consecutive nonnegative 
% integers as classification labels, i.e., 0,1,...,n-1, where n is the 
% number of classes.
% 
% The test creates random data for training data matrix D, of size m = mpos
% + mneg data points (rows of matrix), given as input. The classification 
% problem is set up to ensure that an optimal dividing hyperplane x will
% be close to satisying x_1 = x_2 (points above this line are labelled +1,
% points below are labelled -1 in the training data). The values mpos and 
% mneg are the number of +1 and -1 labelled data points (rows) in D, 
% respectively. The seed specifies the seed value to use in random number 
% generation, for repeatability. The value errtol is the relative error 
% tolerance from the 'true' solution that is allowed for the test to be
% successful. The variable quiet determines whether to suppress output and
% graphs. The parameter sep specifies the difficulty in linear separability
% of the generated problem, and is a positive value between 0 and 1.
%
% The test is run on both hinge and 0-1 loss functions (transpose reduction
% allows 0-1 loss function to be used). You can set a different
% regularization parameter C to use by options.C = (positive decimal
% value). The default value is 0.5.
%
% Consult the user manual for instructions and examples on how to set the
% options argument to customize the solver.
%
% NOTE: If the linearsvmtest function is executed with no inputs, it will
% generate a random problem of size m = 2^8, n = 2, with two classes 
% labelled (+1, -1), where +1 means a point x is above the x_1 = x_2 line,
% and -1 below. The data is generated to be linearly separable in this way.
% For label +1, 2^7 random points are generated, and the same for label -1.

function [results, test] = ...
    linearsvmtest(seed, mpos, mneg, sep, errtol, quiet, options)
% INPUTS ------------------------------------------------------------------
% seed:     The seed used to generate random data in the test.
% mpos:     The number of +1 labelled rows in matrix D. The number of rows
%           in D is equal to m = mpos + mneg.
% mneg:     The number of -1 labelled rows in matrix D. The number of rows
%           in D is equal to m = mpos + mneg.
% sep:      The difficulty in linear separability of the problem. A
%           parameter between 0 and 1, where values approaching 1 approach
%           highest difficulty.
% errtol:   The relative error from the 'true' solution that is allowed for
%           the test to be considered successfully convergent. Typical
%           values are 0.05 to 0.01, corresponding to 5% or 1% accuracy
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
%           interpreter after execution. In this function, results will
%           contain fields hingeloss and zoloss, which contain results for
%           the hinge loss and 0-1 loss functions, respectively.
% test:     A struct containing the details of the random test that was
%           performed, including the random data generated, how correct it
%           was, whether it reached the correct tolerance (successful test)
%           and so on. Consult the user manual for more details on the test
%           struct, or simply check what the variable contains in the 
%           Matlab interpreter after execution. In this function, test will
%           contain fields hingeloss and zoloss, which contain the test 
%           setup for the hinge loss and 0-1 loss functions, respectively.
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
    mpos = 2^7;                 % Number of +1 labeled elements.
    mneg = 2^7;                 % Number of -1 labeled elements.
    errtol = 0.05;
    quiet = 0;
    sep = 0.2;
    C = 0.5;
    options = struct();
else
    % Error checking on input.
    [seed, mpos, mneg, sep, errtol, quiet, C] = ...
        errorcheck(seed, mpos, mneg, sep, errtol, quiet, options);

    rng('default');             % Default RNG.
    rng(seed);                  % Seed the RNG.
end

m = mpos + mneg;                    % Total number of labeled elements.

% Build random positive and negative elements for which the dividing
% solution is guaranteed to be near x_1 = x_2 for ADMM output x.
pos = [(0:2/(mpos - 1):2)' + rand(mpos, 1) - sep*rand(mpos, 1), ...
    (0:2/(mpos - 1):2)' - rand(mpos, 1) + sep*rand(mpos, 1)];
neg = [(0:2/(mpos - 1):2)' - rand(mneg, 1) + sep*rand(mneg, 1), ...
    (0:2/(mpos - 1):2)' + rand(mneg, 1) - sep*rand(mneg, 1)];

D = [pos; neg];                     % Vector of grouped elements.
ell = ones(m, 1);                   % Labels for each element (+1 or -1).
ell(mpos + 1:end) = -1;             % Set negative labels.


% Set the options struct for ADMM.
options.objevals = 1;
options.convtest = 1;
options.tester = 'linearsvm';

% Theoretically, the true optimal objective value will be at x = [1; -1].
trueobjopt = 1/2*norm([1;-1], 'fro')^2 + ...
    C*sum(max(sign(1 - ell.*(D*[1;-1])),0));

% Run the Linear SVM solver on this set up with hinge loss function.
resultsh = linearsvm(D, ell, C, options);

% Run Linear SVM solver once again with 0-1 loss fuction this time.
options.lossfunction = '0-1';
results01 = linearsvm(D, ell, C, options);

% ADMM results for optimal x and objective value for hinge loss.
xopth = resultsh.xopt;
admmopth = resultsh.objopt;
objopth = resultsh.options.obj(xopth, xopth);

% ADMM results for optimal x and objective value for 0-1 loss.
xopt01 = results01.xopt;
admmopt01 = results01.objopt;
objopt01 = results01.options.obj(xopt01, xopt01);

% Relative error from true solution for x_1 component for both loss
% functions. The error should be close to 0.
relerrorh = abs(1 - (-xopth(2)/xopth(1)));
relerror01 = abs(1 - (-xopt01(2)/xopt01(1)));

% Check to see if the test was successful or not (within error tolerance)
% for hinge loss.
if (objopth < trueobjopt && relerrorh <= errtol)
    failedh = 0;
else
    failedh = 1;
end

% Check to see if the test was successful or not (within error tolerance)
% for 0-1 loss function.
if (objopt01 < trueobjopt && relerror01 <= errtol)
    failed01 = 0;
else
    failed01 = 1;
end

% Populate the test struct with details about the test and its results for
% the hinge loss function.
testh.truexopt = [1; -1];
testh.trueobjopt = trueobjopt;
testh.xopt = xopth;
testh.admmopt = admmopth;
testh.objopt = objopth;
testh.failed = failedh;
testh.relerror = relerrorh;
testh.xresidual = norm(xopth - [1; -1], 'fro');
testh.steps = resultsh.steps;
testh.errtol = errtol;

% Populate the test struct with details about the test and its results for
% the 0-1 loss function.
test01.truexopt = [1; -1];
test01.trueobjopt = trueobjopt;
test01.xopt = xopt01;
test01.admmopt = admmopt01;
test01.objopt = objopt01;
test01.failed = failed01;
test01.relerror = relerror01;
test01.xresidual =  norm(xopt01 - [1; -1], 'fro');
test01.steps = results01.steps;
test01.errtol = errtol;

% Populate the test output struct with the results of both types of loss
% functions.
test.hingeloss = testh;
test.zoloss = test01;

% Populate the results output struct with the results of both types of loss
% functions.
results.hingeloss = resultsh;
results.zoloss = results01;

% Show plots and results if user didn't specify to be quiet.
if ~quiet

    % Show general plots for results for both loss functions.
    options.lossfunction = 'hinge';
    showresults(results.hingeloss, test.hingeloss, options);
    options.lossfunction = '01';
    showresults(results.zoloss, test.zoloss, options);
    
    % Linear form solutions for x_1 component for the hinge and 0-1 loss.
    x1funh = @(x2) -xopth(2)/xopth(1)*x2;
    x1fun01 = @(x2) -xopt01(2)/xopt01(1)*x2;

    x2 = -1:1/1000:3;               % x_2 vector to evaluate over for plot.
    xh_1 = x1funh(x2);              % x_1 vector for x_2 values for hinge.
    x01_1 = x1fun01(x2);            % x_1 vector for x_2 values for hinge.

    % Figure plotting the positive (red) and negative (blue) labeled
    % elements and the best dividing line solutions computed via ADMM.
    figure;
    plot(x2, xh_1, 'b', x2, x01_1, 'm', pos(:, 2), pos(:, 1), 'ro', ...
        neg(:, 2), neg(:, 1), 'go');
    ylabel('x_1');
    xlabel('x_2');
    title('Linear SVM Test For Hinge And 0-1 Loss Functions');
    legend('Hinge Solution', '0-1 Solution', 'Positive', 'Negative');

    % Display some pertinent information about the results.
    disp(' ');
    fprintf('Hinge Loss Hyperplane: \t\t%d*x_1 + %d*x_2 = 0\n', ...
        xopth(1), xopth(2));
    fprintf('Hinge Loss Standard Form: \tx_1 = %d*x_2\n\n', ...
        -xopth(2)/xopth(1));
    fprintf('0-1 Loss Hyperplane: \t\t%d*x_1 + %d*x_2 = 0\n', ...
        xopt01(1), xopt01(2));
    fprintf('0-1 Loss Standard Form: \tx_1 = %d*x_2\n\n', ...
        -xopt01(2)/xopt01(1));
    display('Theoretical result should be x_1 = x_2.');

end

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Simply checks for any user error on same input, stops if error is
% detected. If input is correct, return info needed for problem.

function [seed, mpos, mneg, sep, errtol, quiet, C] = ...
    errorcheck(seed, mpos, mneg, sep, errtol, quiet, options)
% INPUTS ------------------------------------------------------------------
% Same as for the linearsvmtest function.
% 
% OUTPUTS -----------------------------------------------------------------
% Same as the inputs, except that C is the regularization parameter
% obtained from options struct, i.e., 'options.C'.
% -------------------------------------------------------------------------


% Check for invalid input and report errors / terminate if detected.
if (~isnumeric(seed) || floor(seed) < 0)
    error('Given RNG seed is not a nonnegative integer!');
elseif (~isnumeric(mpos) || floor(real(mpos)) < 0)
    error(['Given number mpos of +1 labelled data points is not a', ...
        ' nonnegative integer!']);
elseif (~isnumeric(mneg) || floor(real(mneg)) < 0)
    error(['Given number mneg of -1 labelled data points is not a', ...
        ' nonnegative integer!']);
elseif (~isnumeric(sep) || real(sep) < 0 || real(sep) > 1)
    error(['Given seperability parameter is invalid! Must be between ', ...
        '0 and 1!']);
elseif (~isnumeric(errtol) || real(errtol) <= 0)
    error('Given error tolerance errtol is not a positive number!');
else
    seed = floor(real(seed));   % Set the seed to valid value.
    mpos = floor(real(mpos));   % Set the number of +1s to valid value.
    mneg = floor(real(mneg));   % Set the number of -1s to valid value.
    sep = real(sep);            % Set the separability parameter.
    errtol = real(errtol);      % Set error tolerance to valid value.
end

% Only set output to be suppressed if specified.
if (~isnumeric(quiet) || floor(real(quiet)) ~= 1)
    quiet = 0;
else
    quiet = 1;
end

% Check that the given options is actually a struct...
if ~isstruct(options)
    error('Given options is not a struct! At least pass empty struct!');
end

% Set regularization parameter, if specified.
if isfield(options, 'C')
    C = options.C;                  % Take user specified value.
    
    % Report if there are any issues with the regularization parameter.
    if (~isnumeric(C) || real(C) < 0)
        error('Given regularization parameter C is an invalid value!');
    else
        C = real(C);
    end
else
    C = 0.5;                        % Set default regularization parameter.
end

end
% -------------------------------------------------------------------------