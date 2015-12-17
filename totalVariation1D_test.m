rng(1);

test_correctness = 0;
test_adaptive_correctness = 0;
showgraphs = 0;
min_power = 5;
max_power = 10;
max_tests = 1;
max_cvx_size = 7;
max_iter = 300;
rho = 1.0;
alpha = 1.0;
lambda = 5.0;
errtol = 1e-1;
ave_time = zeros(1, max_power - min_power + 1);
sizes = zeros(1, max_power - min_power + 1);

for i = min_power:max_power
    
    index = i - min_power + 1;
    
    for j = 1:max_tests;
        
        m = 2^i;
        n = m;
        
        x0 = ones(n, 1);                % Vector to store sampling.

        % Purpose: Populate x0 with random data.
        for k = 1:3

            rs = randsample(n, 1);      % Random integer from 1 to n.
            ri = randsample(1:10, 1);   % Random integer from 1 to 10..

            % Assign values to x0 in random spots.
            x0(ceil(rs/2):rs) = ri*x0(ceil(rs/2):rs);

        end

        signal = x0 + randn(n, 1);           % Add small random numbers.

        tic;
        [objval, x, ~, iter, objvalS, xS, ~, iterS] = ...
            totalVariation1D(signal, lambda, rho, alpha, max_iter, 1);
        ave_time(index) = ave_time(index) + toc;
        
        disp(['Adaptive: ', num2str(length(iter.objval)), ... 
            ' iterations, vs. Static: ', num2str(length(iterS.objval))]);
        
        if (test_adaptive_correctness)
            assert(abs((objvalS - objval)/objval) <= errtol, ...
                strcat('objvalS: %d =/= objvalA: %d, absdiff:', ...
                ' %d > %d, for n = %i, j = %i'), objvalS, objval, ...
                abs((objvalS - objval)/objval), errtol, n, j)
        end
        
        if (j == 1 && showgraphs)
            
            N = length(iter.objval);
            
            % Figure for original signal vs. denoised signal.
            figure;
            plot(1:n, signal, 'k', 1:n, x, 'red', 'MarkerSize', 10, 'LineWidth', 2);
            title('Original Signal (black) and Denoised Signal (red)');
            ylabel('Signal value');
            xlabel('Time step');

            % Figure for the objective value at each iteration.
            figure;
            subplot(3, 1, 1);
            plot(1:N, iter.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
            title('Plot of objective value for each iteration');
            ylabel('Objective'); 
            xlabel('Iteration k');
            
            % Figure of primal residual norm.
            subplot(3, 1, 2);
            semilogy(1:N, max(1e-8, iter.pnorm), 'k', ...
                1:N, iter.perr, 'k--',  'LineWidth', 2);
            title('Plot of Primal Residual Norm');
            ylabel('||Ax + Bz - c||_2');
            xlabel('Iteration k');
            
            % Figure of dual residual norm
            subplot(3, 1, 3);
            semilogy(1:N, max(1e-8, iter.dnorm), 'k', ...
                1:N, iter.derr, 'k--', 'LineWidth', 2);
            title('Plot of Dual Residual Norm');
            ylabel('||rho*A^TB(z^k-z^{k-1})||_2'); 
            xlabel('Iteration k');
            
        end
        
        if (test_correctness && i <= max_cvx_size)
            cvx_begin quiet

                variable u(n);
                minimize(1/2*pow_pos(norm(u - signal, 2), 2) + ...
                    lambda*sum(abs(u(2:length(u)) - u(1:length(u)-1))));

            cvx_end

            if (isnan(cvx_optval))
                disp(['CVX failed for: n = ', num2str(2^i), ', j = ', ...
                    num2str(j)]);
                continue;
            end

            assert(abs((objval - cvx_optval)/cvx_optval) <= errtol, ...
                strcat('objval: %d =/= cvx_optval: %d, absdiff:', ...
                ' %d > %d, for n = %i, j = %i'), objval, cvx_optval, ...
                abs((objval - cvx_optval)/cvx_optval), errtol, n, j)

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
