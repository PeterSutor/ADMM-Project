function results = unwrappedadmm(zming, D, options)

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

tic;                                            % Start timing.

% If no arguments are given, run a demo test by running the linearsvmtest
% function with no arguments. Creates a test as described in the NOTE above
% in the description.
if nargin == 0
    display(['No arguments detected; running a demo test of the', ...
        ' Linear SVM problem for random data of size m = 2^8, n = 2:']);
    display(['2 classes with labels +1 and -1; for data point x,', ...
        ' +1 means it is above the x_1 = x_2 axis, -1 means below:']);
    options.parallel = 'both';
    results = linearsvmtest(0, 2^7, 2^7, 0.2, 0.05, 0, options);
    results.solverruntime = toc;                % End timing.
    
    return;
end

[m, n] = size(D);

if isfield(options, 'parallel') && (strcmp(options.parallel, 'xminf') ||...
    strcmp(options.parallel, 'zming') || strcmp(options.parallel, 'both'))
    pool = gcp;
    
    if (strcmp(options.parallel, 'both'))
        options.parallel = 'zming';
    else
        options.parallel = 'none';
    end
    
    xminf = @proxf;
    options.preprocess = @nodepreprocessing;
    sliceranges = 0;
    Di = 0;
    Dti = 0;
    Wi = 0;
    W = 0;
    di = 0;
    
    if isfield(options, 'slices')
        errorcheck(options.slices, 'isvector', 'options.slices');
        options.slices = real(floor(options.slices));
    else
        options.slices = 0;
    end
    
    sliceopts.workers = pool.NumWorkers;
    sliceopts.slicelength = size(D, 1);
    options.slices = errorcheck(options.slices, 'slices', ...
        'options.slices', sliceopts);
else
    Dplus = pinv(D);
    
    xminf = @(x,z,u,rho) Dplus*(z - u);
end

options.A = D;
options.At = D';
options.B = -1;
options.nB = m;
options.c = 0;
options.m = m;
options.x0 = rand(n, 1);
options.z0 = rand(m, 1);
options.u0 = rand(m, 1);
options.maxiters = 1000;
options.stopcond = 'both';
options.nodualerror = 1;

results = admm(xminf, zming, options);

    function nodepreprocessing()

        slicenum = length(options.slices);
        runningsum = 0;
        sliceranges = cell(slicenum, 1);
        Di = cell(slicenum, 1);
        Dti = cell(slicenum, 1);
        di = cell(slicenum, 1);

        for i = 1:slicenum
            sliceranges{i} = [runningsum+1, runningsum+options.slices(i)];
            Di{i} = D(runningsum+1:runningsum+options.slices(i), :);
            Dti{i} = Di{i}';
            runningsum = runningsum + options.slices(i);
        end

        Wi = cell(slicenum, 1);

        parfor i = 1:slicenum
            Wi{i} = Dti{i}*Di{i};
        end

        W = zeros(size(Wi{1}));

        for i = 1:slicenum
            W = W + Wi{i};
        end
    end

    function xmin = proxf(~, z, u, ~)
    
        parfor i = 1:length(options.slices)
            slicez = z(sliceranges{i}(1):sliceranges{i}(2)); %#ok<PFBNS>
            sliceu = u(sliceranges{i}(1):sliceranges{i}(2)); %#ok<PFBNS>
            di{i} = Dti{i}*(slicez - sliceu);
        end
        
        d = 0;
        
        parfor i = 1:length(options.slices)
            d = d + di{i};
        end
        
        xmin = W \ d;
        
    end

end