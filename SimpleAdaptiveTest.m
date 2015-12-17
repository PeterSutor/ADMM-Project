rng(2);

n = 500;
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
proxf = @(x, z, u, rho) (AtA2 + rho*I) \ (Atb2 + rho*(z - u));
proxg = @(x, z, u, rho) (CtC2 + rho*I) \ (Ctd2 + rho*(x + u));
ddf = @(x, z, u, rho) -A*inv(A)'*(A\u + b);

[objvalA, xA, zA, iterA] = adaptive_admm(obj, proxf, proxg, ddf, ...
    rho, 1, 1, -1, zeros(n, 1), 1000, 1);

options.obj = obj;
options.rho = rho;
options.A = eye(n, n);
options.At = options.A';
options.B = -options.A;
options.c = zeros(n, 1);
options.maxiters = 1000;

[results] = admm(proxf, proxg, options);

objvalS = results.objopt;
xS = results.xopt;

% cvx_begin quiet
% 
%     variable u(n);
%     minimize(pow_pos(norm(A*u - b, 2), 2) + pow_pos(norm(C*u - d, 2), 2));
% 
% cvx_end

x = (AtA + CtC) \ (Atb + Ctd);
objx = obj(x, x);

display('______________________________________________________________');
disp(['Actual Solution: ', num2str(objx)]);
disp(['Objective vals (Adaptive vs. Static): ', num2str(objvalA), ...
    ' vs. ', num2str(objvalS)]);
disp(['Iterations (Adaptive vs. Static): ', ...
    num2str(length(iterA.objval)), ' vs. ', ...
    num2str(results.steps)]);