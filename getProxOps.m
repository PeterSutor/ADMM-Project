function [minx, minz] = getProxOps(problem, args)

switch(problem)
    case 'Model'
        AtA2 = args.AtA2;
        Atb2 = args.Atb2;
        CtC2 = args.CtC2;
        Ctd2 = args.Ctd2;
        n = args.n;
        minx = @xminModel;
        minz = @zminModel;
    case 'BasisPursuit'
        P = args.P;
        q = args.q;
        minx = @xminBasisPursuit;
        minz = @zminBasisPursuit;
    case 'TotalVariation1D'
        Id = args.Id;
        D = args.D;
        Dt = args.Dt;
        DtD = args.DtD;
        signal = args.signal;
        alpha = args.alpha;
        lambda = args.lambda;
        minx = @xminTotalVariation1D;
        minz = @zminTotalVariation1D;
    case 'LinearSVM'
        Id = args.Id;
        D = args.D;
        Dt = args.Dt;
        Dplus = args.Dplus;
        ell = args.ell;
        C = args.C;
        hinge = args.hinge;
        minx = @xminLinearSVM;
        minz = @zminLinearSVM;
    case 'LASSO'
        A = args.A;
        Atb = args.Atb;
        L = args.L;
        U = args.U;
        m = args.m;
        n = args.n;
        alpha = args.alpha;
        lambda = args.lambda;
        minx = @xminLASSO;
        minz = @zminLASSO;
end


function [minx] = xminModel(~, z, u, rho)
    AtA2new = AtA2;
    AtA2new(1:n+1:end) = AtA2new(1:n+1:end) + rho;
    minx = AtA2new \ (Atb2 + rho*(z - u));
    %minx = (AtA2 + rho) \ (Atb2 + rho*(z - u));
end

function [minz] = zminModel(x, ~, u, rho)
    CtC2new = CtC2;
    CtC2new(1:n+1:end) = CtC2new(1:n+1:end) + rho;
    minz = CtC2new \ (Ctd2 + rho*(x + u));
end

function [minx] = xminBasisPursuit(~, z, u, ~) 
    minx = P*(z - u) + q;
end

function [minz] = zminBasisPursuit(x, ~, u, rho) 
    minz = sign(u - x).*subplus(abs(u - x) - 1/rho);
end

function [minx] = xminTotalVariation1D(~, z, u, rho)
    minx = (Id + rho*DtD) \ (signal + rho*Dt*(z - u));
end

function [minz] = zminTotalVariation1D(x, z, u, rho)
    zprev = z;
    Ax = alpha*D*x + (1 - alpha)*zprev;
    v = u + Ax;
    minz = sign(v).*subplus(abs(v) - lambda/rho);
end

function [minx] = xminLinearSVM(~, z, u, ~)
    minx = Dplus*(z - u);                       % The x-minimization step.
end

function [minz] = zminLinearSVM(x, ~, u, rho)
    Dx = D*x;                                   % Save this product.
    v = ell.*(Dx + u);                          % Save label product.
    
    % The z-minimization step, based on whether to use Hinge or 0-1 loss
    % function.
    if hinge
        minz = Dx + u + ell.*max(min(1 - v, C/rho), 0);
    else
        minz = ell.*minz01(v, rho/C);
    end
    
    % -------------------------------------------------------------------------
    % Prox for 0-1. Minimize z(y) + 1/2t||y - s||^2.
    function y = minz01(s, t)
    % INPUTS: 
    % s     Vector to use in proximal operator.
    % t     Proximal parameter.
    % -------------------------------------------------------------------------
    % OUTPUTS:
    % y     Result of proximal operator.
    % -------------------------------------------------------------------------

    % Initialization.
    y = ones(length(s), 1);

    % Compute indicaes where label would be classified as 1 and isolate those
    % indices from s in y.
    inds = (s >= 1) | (s < (1 - sqrt(2/t)));
    y(inds) = s(inds);

    end
end

function [minx] = xminLASSO(~, z, u, rho)
% Performs our x-minimization step. Acts as proximal operator.
    
    y = rho*(z - u) + Atb;     % For ease of writing.

    % Solve the system efficiently based on whether A was short or tall.
    if(m >= n)                  % A is square or tall.
        % Solve system via LU-decomposition efficiently.
        minx = U \ (L \ y);
    else                        % A is short and fat.
        % Solve via LU-decomposition with roles of A and A transpose
        %   swapped. We get a solution of length n, as desired.
        minx = 1/rho*y - 1/rho^2*(A'*(U \ (L \ (A*y))));
    end
end

function [minz] = zminLASSO(x, z, u, rho)
    zprev = z;
    xprime = alpha*x + (1 - alpha)*zprev;
    v = u + xprime;
    minz = sign(v).*subplus(abs(v) - lambda/rho);
end

end

