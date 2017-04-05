function  unwrappedadmmtest(minscale, maxscale, trials, options)

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


options.parallel = 'both';
options.scaler = '';
r1 = solvertester('linearsvm', minscale, maxscale, trials, 0, options);

options.parallel = 'none';
r2 = solvertester('linearsvm', minscale, maxscale, trials, 0, options);

sizes = power(2, minscale:maxscale);

figure
semilogy(sizes, r1.avetimes(:, 1, 1), '-ro', sizes, ...
    r2.avetimes(:, 1, 1), '-bo')
legend('Two-core Parallel Linear SVM', 'Two-core Regular Linear SVM')
legend('Two-core Parallel Linear SVM', 'Two-core Regular Linear SVM', ...
    'Location', 'south')
xlabel('Number of rows in D');
ylabel('Time (seconds)');
title(['Runtime Comparison of Regular Unwrapped ADMM vs. Parallel', ...
    ' With Transpose Reduction']);

figure
loglog(sizes, r1.avetimes(:, 1, 1), '-ro', sizes, ...
    r2.avetimes(:, 1, 1), '-bo')
legend('Two-core Parallel Linear SVM', 'Two-core Regular Linear SVM')
legend('Two-core Parallel Linear SVM', 'Two-core Regular Linear SVM', ...
    'Location', 'south')
xlabel('Number of rows in D');
ylabel('Time (seconds)');
title(['Runtime Comparison of Regular Unwrapped ADMM vs. Parallel', ...
    ' With Transpose Reduction']);

end

