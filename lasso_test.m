rng('default');
rng(1);

test_correctness = 0;
max_power = 9;
min_power = 5;
max_tests = 100;
max_cvx_size = 7;
rho = 0.5;
relax = 1.0;
errtol = 1e-2;
ave_time = zeros(1, max_power - min_power + 1);
sizes = zeros(1, max_power - min_power + 1);

for i = min_power:max_power
    
    index = i - min_power + 1;
    
    for j = 1:max_tests;
        
        n = 2^i;
        m = 100;

        d = 1/50;                       % Sparsity density.

        %--------------------------------------------------------------------------
        % Generates our matrix A and vector b in Lasso Problem.
        x0 = sprandn(n, 1, d);          % Sparse, random, normally distributed.
        A = randn(m, n);                % Just a random matrix.

        % Z-score A as usual with data that is to be normalized.
        A = A*spdiags(1./sqrt(sum(A.^2))', 0, n, n);

        % Z-scored vector b with perturbations to minimize.
        b = A*x0 + sqrt(0.001)*randn(m, 1);
        %--------------------------------------------------------------------------

        lambda_max = norm(A'*b, 'inf');
        lambda = 0.1*lambda_max;
        
        options.objevals = 1;
        options.rho = rho;
        options.maxiters = 1000;
        options.relax = relax;
                                        
        tic;
        results = lasso(A, b, lambda, options);
        ave_time(index) = ave_time(index) + toc;
        
        objval = results.objopt;
        
        if (j == 1)
            
            N = results.steps;

            % Figure for the objective value at each iteration.
            figure;
            subplot(3, 1, 1);
            plot(1:N, results.objevals, 'k', 'MarkerSize', 10, 'LineWidth', 2);
            title('Plot of objective value for each iteration');
            ylabel('Objective'); 
            xlabel('Iteration k');
            
            % Figure of primal residual norm.
            subplot(3, 1, 2);
            semilogy(1:N, max(1e-8, results.pnorm), 'k', ...
                1:N, results.perr, 'k--',  'LineWidth', 2);
            title('Plot of Primal Residual Norm');
            ylabel('||Ax + Bz - c||_2');
            xlabel('Iteration k');
            
            % Figure of dual residual norm
            subplot(3, 1, 3);
            semilogy(1:N, max(1e-8, results.dnorm), 'k', ...
                1:N, results.derr, 'k--', 'LineWidth', 2);
            title('Plot of Dual Residual Norm');
            ylabel('||rho*A^TB(z^k-z^{k-1})||_2'); 
            xlabel('Iteration k');
            
        end
        
        if (test_correctness && i <= max_cvx_size)
            cvx_begin quiet

                variable u(n);
                minimize(0.5*sum((A*u - b).^2) + lambda*norm(u, 1));

            cvx_end

            if (isnan(cvx_optval))
                disp(['CVX failed for: n = ', num2str(2^i), ', j = ', ...
                    num2str(j)]);
                continue;
            end

            assert(abs((objval - cvx_optval)/cvx_optval) <= errtol, ...
                strcat('objval: %d =/= cvx_optval: %d, absdiff:', ...
                ' %d > %d, for n = %i'), objval, cvx_optval, ...
                abs((objval - cvx_optval)/cvx_optval), errtol, n)
        end

    end
    
    ave_time(index) = ave_time(index) / max_tests;
    sizes(index) = 2^i;
    disp(['Average time for size 2^', num2str(i), ': ', ...
        num2str(ave_time(index)), ' seconds.']);
    
end

if (test_correctness)
    disp(['Error within expected tolerances for', ' input size 2^', ...
        num2str(max_cvx_size), ' and smaller!']);
end

figure;
loglog(sizes, ave_time);
title('Log-log plot of average ADMM run-time over size 2^x');
xlabel('Order of matrix and vector sizes (size 2^x)');
ylabel('Average run-time in seconds');
