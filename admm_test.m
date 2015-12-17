rng('default');
rng(2);

test_correctness = 1;
showgraphs = 1;
max_power = 9;
min_power = 4;
max_tests = 3;
rho = 5.0;
iterations = 1000;
errtol = 1e-2;
ave_time = zeros(1, max_power - min_power + 1);
sizes = zeros(1, max_power - min_power + 1);

incorrect = 0;

for i = min_power:max_power
    for j = 1:max_tests;
        
        n = 2^i;
        rho = 5.0;

        A = rand(n, n);
        At = A';
        AtA = At*A;
        AtA2 = 2*AtA;
        C = rand(n, n);
        Ct = C';
        CtC = Ct*C;
        CtC2 = 2*CtC;
        b = rand(n, 1);
        Atb = At*b;
        Atb2 = 2*Atb;
        d = rand(n, 1);
        Ctd = Ct*d;
        Ctd2 = 2*Ctd;
        I = eye(n, n);
        
        obj = @(x, z) norm(A*x - b, 2)^2 + norm(C*z - d, 2)^2;
        
        args.AtA2 = AtA2;
        args.Atb2 = Atb2;
        args.CtC2 = CtC2;
        args.Ctd2 = Ctd2;
        args.n = n;
        
        [proxf, proxg] = getProxOps('Model', args);
        
        options = struct;
        options.obj = obj;
        options.rho = rho;
        options.maxiters = iterations;
        options.c = zeros(n, 1);
        options.A = eye(n, n);
        options.At = eye(n, n);
        options.B = -eye(n, n);
        options.objevals = 1;
        options.convtest = 1;
        options.convtol = 1e-3;
        options.stopcond = 'both';
        
        tic;

        [results] = admm(proxf, proxg, options);
        
        objval = obj(results.xopt, results.xopt);
        
        ave_time(i - min_power + 1) = ave_time(i - min_power + 1) + toc;
        
        if (j == 1 && showgraphs)
            
            N = results.steps;

            % Figure for the objective value at each iteration.
            figure;
            subplot(3, 1, 1);
            plot(1:N, results.objopt, 'k', 'LineWidth', 2);
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
        
        if (test_correctness)
            x = (AtA + CtC) \ (Atb + Ctd);
            objx = obj(x, x);
            
            if (abs((objval - objx)/objx) <= errtol)
                fprintf('For n = 2^%i, test %i -- Relative error acceptable: %d\n', ...
                    i, j, abs((objval - objx)/objx));
            else
                fprintf('For n = 2^%i, test %i -- RELATIVE ERROR UNACCEPTABLE: %d; %d vs. true %d\n', ...
                    i, j, abs((objval - objx)/objx), objval, objx);
                incorrect = incorrect + 1;
            end
            
        end

    end
    
    ave_time(i - min_power + 1) = ave_time(i - min_power + 1) / max_tests;
    sizes(i - min_power + 1) = 2^i;
    
end

for i = min_power:max_power
    disp(['Average time for size 2^', num2str(i), ': ', ...
        num2str(ave_time(i - min_power + 1)), ' seconds.']);
end

if (test_correctness && incorrect == 0)
    disp(['Error within expected relative tolerances (', num2str(errtol), ') for', ' input sizes 2^', ...
        num2str(min_power), ' up to 2^', num2str(max_power)]);
elseif (test_correctness && incorrect ~= 0)
    disp([num2str(incorrect), ' UNACCEPTABLE ERROR(S) FOR TOLERANCE ', ...
        num2str(errtol), ', for ', num2str(iterations), ' iterations!']);
end

if (showgraphs)
    figure;
    loglog(sizes, ave_time);
    title('Log-log plot of average ADMM run-time over size 2^x');
    xlabel('Order of matrix and vector sizes (size 2^x)');
    ylabel('Average run-time in seconds');
end
