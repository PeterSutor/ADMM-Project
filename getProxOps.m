% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION
%
% Returns the proximal operators for the solver specified. Constructs the
% proximal operators and caches any important values for every solver in
% the library.
%
% Consult the user manual for instructions and examples on how to set the
% args argument.

function [minx, minz, extra] = getproxops(problem, args)
% INPUTS ------------------------------------------------------------------
% problem:  A character string specifying the solver to use. See the switch
%           statement below to see valid values for this input.
% args:     A struct specifying the arguments needed for creating the
%           proximal operators. See the corresponding case in the switch to
%           see which arguments are needed for each problem.
% 
% OUTPUTS -----------------------------------------------------------------
% minx:     The proximal operator function minimizing for x the Augmented
%           Lagrangian for ADMM for the given problem. Inputs are x, z, u,
%           and step size rho.
% minz:     The proximal operator function minimizing for z the Augmented
%           Lagrangian for ADMM for the given problem. Inputs are x, z, u,
%           and step size rho.
% -------------------------------------------------------------------------


extra = struct();

% Error checking ----------------------------------------------------------
% -------------------------------------------------------------------------
% If problem is a string, return lowercase of it, else return error.
if ischar(problem)
    problem = lower(problem);
else
    error(['Given problem argument is not a string specifying for ', ...
        'which problem proximal operators are needed!']);
end

% If args is not a struct, return error.
if ~isstruct(args)
    error(['Given struct args is not a struct containing arguments ', ...
        'needed for proximal operators for the given problem!']);
end
% -------------------------------------------------------------------------

% Switch statement on problem to determine which problem to return proximal
% operators for.
switch(problem)
    % ---------------------------------------------------------------------
    % The Model Problem (see below) ---------------------------------------
    case 'model'
        % Returns proximal operators for solving a simple model problem 
        % using ADMM. The model problem to solve is minimizing for x the 
        % objective function:
        %    obj(x) = 1/2*(||P*x - r||_2)^2 + 1/2*(||Q*x - s||_2)^2,
        % where P is an m by n matrix, Q is an m by n matrix, and r and s
        % are column vectors of length m. Thus, x is a column vector of
        % length n.
        % 
        % The ADMM Augmented Lagrangian here is:
        %    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + 
        %                   constant(u),
        % where f(x) = 1/2*(||P*x - r||_2)^2 and ...
        % g(z) = 1/2*(||Q*z - s||_2)^2, i.e. such that ... 
        % obj(x) = f(x) + g(x). The rho in L_rho is the dual step size 
        % parameter. ADMM solves the model problem by minimizing: 
        % f(x) + g(z), subject to x - z = 0.
        %
        % The proximal operator for f is the result of taking an x 
        % derivative of the model's L_rho and solving it for 0, i.e.:
        %    P^T*(P*x - r) + rho*(x - z + u) = 0
        % solved for x. This involves solving a simple linear system.
        %
        % The proximal operator is g the result of taking a z derivative of
        % the model's L_rho and solving it for 0, i.e.:
        %    Q^T*(Q*z - s) - rho*(x - z + u) = 0
        % solved for z. This involves solving a simple linear system.
        
        PtP = args.PtP; % The matrix P^T*P for P above.
        PtPnew = PtP;   % The future PtP that will be perturbed by rho.
        Ptr = args.Ptr; % The vector P^T*r for P and r above.
        QtQ = args.QtQ; % The matrix Q^TQ for Q above.
        QtQnew = QtQ;   % The future QtQ that will be perturbed by rho.
        Qts = args.Qts; % The vector Q^T*s for Q and s above.
        n = args.n;     % The number of columns n for the m by n problem.
        rhoprev = 0;    % The previous rho used. Determines whether P and Q
                        % need to get reconstructed.
        
        % Return the appropriate proximal operators for minimizing x and z.
        minx = @xminModel;
        minz = @zminModel;
    % ---------------------------------------------------------------------
    % The Basis Pursuit Problem (see below) -------------------------------
    case 'basispursuit'
        % Returns the proximal operators for solving the Basis Pursuit 
        % problem using ADMM. Basis Pursuit minimizes for x the objective 
        % function:
        %    obj(x) = ||x||_1, 
        %    subject to D*x = s,
        % where D is a matrix and s is a column vector of appropriate
        % length. Thus, x is a column vector of the same length.
        % 
        % The ADMM Augmented Lagrangian here is:
        %    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + 
        %                   constant(u),
        % where f(x) is the indicator function that x is not in the set 
        % {x: D*x = s} and g(z) = ||z||_1. The rho in L_rho is the dual
        % step size parameter. ADMM solves the equivalent problem of
        % minimizing: f(x) + g(z), subject to x - z = 0.
        %
        % Cache the following matrix: P = I - D^T*(D*D^T)^-1*D, specified
        % by the user. Likewise, cache q = D^T*(D*D^T)^-1*s. A projection
        % of v onto the set {x: D*x = s} is then x = P*v + q. Note that
        % this collapses into x = D^-1*s if D is square, but otherwise
        % still works despite the dimensions of D; this is the product of s
        % with the Pseudoinverse of D.
        % 
        % The proximal operator for f, which indicates if x is not in the
        % set {x: D*x = s}, reaches its minimum when x IS in the set. 
        % We can guarantee to find such an x by projecting a vector 
        % v = z - u onto this set. This is done using the cached P and q, 
        % by computing the Psuedoinverse of D times v, P*(z - u) + q.
        % 
        % For the proximal operator for g, note that g's formulation is
        % minimizable via soft-thresholding, a proximal mapping technique:
        %    z = min_z(||z - v||_2^2 + lambda*||v||_1) 
        %      = sign(z)*(|z| - lambda/rho)_+ (non positive parts set to 0)
        % This function simply evaluates this for v = u + x, and 
        % appropriate rho. Also note that soft-thresholding typically sets
        % rho = 1, but we move according to the step size parameter rho.
        
        % Cache the matrices P and q, computed by the user, below.
        P = args.P;
        q = args.q;
        
        % Return the appropriate proximal operators for minimizing x and z.
        minx = @xminBasisPursuit;
        minz = @(x, ~, u, rho) zminSoftThresholding(u + x, 1/rho);
    % ---------------------------------------------------------------------
    % The Total Variation Minimizaton Problem (see below) -----------------
    case 'totalvariation'
        % Returns the proximal operators for performing Total Variation 
        % Minimization (TVM) using ADMM. TVM minimizes for x the objective
        % function:
        %    obj(x) = 1/2||x - s||_2^2 + lambda*sum_i{|x_{i+1} - x_i|}, 
        % where x and s are column vectors of length m. The vector s is a
        % given vector of signals to perform TVM on. The minimized vector x
        % is a denoised version of s in the sense of total variation. The
        % constant lambda represents how strictly or loosely to minimize
        % the noise in s.
        % 
        % The ADMM Augmented Lagrangian here is:
        %    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||D*x - z + u||_2)^2 + 
        %                   constant(u),
        % where f(x) = 1/2||x - s||_2^2 and...
        % g(z) = lambda*||z||_1 (by substitution). The rho in L_rho is the
        % dual step size parameter. ADMM solves the equivalent problem of 
        % minimizing: f(x) + g(z), subject to D*x - z = 0. The matrix D is
        % the difference matrix that performs the z_{i+1} - z_i differences
        % for the vector z. Thus, it is a sparse m by m matrix with the
        % stencil [1 -1] along the diagonal entries.
        %
        % For the proximal operator for f, we begin by noting that the
        % gradient of L_rho(x,z,u) is:
        %    grad_x(L_rho(x,z,u)) = grad_x(f(x)) + rho*D^T*(D*x - z + u)
        %                         = (x - s) + rho*D^T*(D*x - z + u)
        % To minimize this, we set it equal to 0 and solve for x, giving:
        %    (x - s) + rho*D^T*(D*x - z + u) = 0 
        %       <--> (I + rho*D^T*D)*x = s + rho*D^T*(z - u)            (1)
        %       <--> x := (I + rho*D^T*D)^-1*(s + rho*D^T*(z - u))
        % Thus, we can solve the system (1) for our minimized x. Note that
        % since I and rho*D^T*D are clearly sparse, we can abuse the 
        % sparsity using sparse matrices.
        %
        % For the proximal operator for g, note that g's formulation is
        % minimizable via soft-thresholding, a proximal mapping technique:
        %    z = min_z(||z - v||_2^2 + lambda*||v||_1) 
        %      = sign(z)*(|z| - lambda/rho)_+ (non positive parts set to 0)
        % This function simply evaluates this for v = u + D*x, and 
        % appropriate rho. Also note that soft-thresholding typically sets
        % rho = 1, but we move according to the step size parameter rho. 
        %
        % Note that in ADMM, the ratio of lambda to rho represents how
        % strictly or loosely to minimize the noise in s.
        
        D = args.D;             % The difference matrix D.
        Dt = args.Dt;           % The transpose of the difference matrix.
        DtD = args.DtD;         % The product D*D^T.
        s = args.s;             % The signal vector s from the TVM problem.
        lambda = args.lambda;   % TVM's lambda parameter.
        Id = speye(size(DtD));  % A sparse identity matrix of size m by m.
        
        % Return the appropriate proximal operators for minimizing x and z.
        minx = @xminTotalVariation;
        minz = @(x, ~, u, rho) zminSoftThresholding(u + D*x, lambda/rho);
    % ---------------------------------------------------------------------
    % Linear Support Vector Machine (SVM) Problem (see below) -------------
    case 'linearsvm'
        % Returns the proximal operators for training Linear Support Vector
        % Machines using ADMM. To do this, we minimize for x the objective
        % function:
        %    obj(x) = 1/2*||x||_2^2 + C*hinge(D*x)
        % where hinge(v) = sum_i{max(1 - ell_i*v_i, 0)} is the hinge loss
        % function, D is vector/matrix of data (each row is a data
        % point/vector) on which we train the SVM, and C is a
        % regularization parameter that specifies whether we use hard
        % linear classification (C = 0) or soft (and to what degree)
        % classification (C > 0), where we allow some points to pass the
        % dividing hyperplane from either class. The vector ell is a vector
        % of classification labels for our classifier; we use consecutive
        % nonnegative integers as classification labels, i.e., 0,1,...,n-1,
        % where n is the number of classes. A label of -1 indicates failure
        % to classify for the class we are training on.
        %
        % The ADMM Augmented Lagrangian here is:
        %    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||D*x - z + u||_2)^2 + 
        %                   constant(u),
        % where f(x) = 1/2||x||_2^2 and g(z) = C*hinge(z). The rho in 
        % L_rho is the dual step size parameter. ADMM solves the equivalent
        % problem of minimizing: f(x) + g(z), subject to D*x - z = 0.
        % Alternatively, Unwrapped ADMM with Transpose Reduction allows the
        % use of the zero-one loss function for g(z), which indicates if z
        % would have been classified incorrectly (thus correct
        % classification is minimizer).
        %
        % To solve this problem, we need to supply ADMM with proximal
        % operators for f and g. We can determine these as follows:
        %
        % For the proximal operator for f, we note that we want to minimize
        % L_rho(x,z,u) for x; as f is differentiable for x, L_rho is as
        % well. We simply set the gradient of L_rho equal to 0 and solve
        % for x to determine the minimizing value:
        %    grad_x(L_rho(x,z,u)) := 0
        %       <--> grad_x(f(x)) + rho*D^T*(D*x - z + u) = 0
        %       <--> x + rho*D^T*(D*x - z + u) = 0
        %       <--> (I + rho*D^T*D)*x = rho*D^T*(z - u)                (1)
        %       <--> x = (I + rho*D^T*D)^-1*[rho*D^T*(z - u)]
        % Thus, we just solve the system in (1) to obtain our minimized x,
        % and this is our proximal operator.
        % 
        % For the proximal operator for g, we recognize that the hinge loss
        % is piecewise differentiable. Thus, we proceed as for function f:
        %    grad_z(L_rho(x,z,u)) := 0
        %       <--> 0 = grad_z(g(z)) - rho*(D*x - z + u)
        %       <--> 0 = 1/rho*grad_z(g(z)) - D*x + z - u
        %       <--> z = (D*x + u) - 1/rho*grad_z(g(z))                 (2)
        % Now note that: 
        %    grad_z(g(z)) = grad_z(C*sum_i{max(1 - ell_i*z_i, 0)})
        %                 = C*sum_i{max(grad_z(1 - ell_i*z_i), 0)}
        %                 = C*sum_i{max(-ell_i*min(1 - ell_i*z_i, 1), 0)}
        % Plugging this into (2) gives:
        % z = (D*x + u) - C/rho*sum_i{max(-ell_i*min(1 - ell_i*z_i, 1), 0)}
        %   = (D*x + u) + ell^T*max(min(1 - ell^T*z, C/rho), 0)
        % This is our proximal operator.
        %
        % Suppose we now want to use the zero-one loss function instead of
        % the hinge function. Normally this is not possible, but thanks to
        % Unwrapped ADMM and Transpose Reduction, this is possible. We use
        % the Pseudoinverse of D to do this, D^+ = (D^T*D)^-1*D^T, which is
        % formed by solving a linear system and is cached. Then the x
        % update from before just becomes D^+*(z - u), our new and improved
        % proiximal operator for f. For g, the zero-one loss proximal 
        % operator seeks to argmin zo(z) + rho/(2*C)||z - v||^2 for z, and
        % zo is the zero-one function, the indicator that z is wrongly
        % classified (minimized when z is correctly classified). Clearly
        % then, zo minimized is whenever v's entries are correctly
        % classified or when v_i < 1 - sqrt(2*C/rho) (this is because
        % Transpose Reduction makes rows independent of each other, thus we
        % can solve simple inequalities to arrive at these conclusions).
        
        % Cache all the info needed for the Transpose Reduction technique.
        D = args.D;              % Cached D matrix from problem above.
        Dt = args.Dt;            % Cached transpose D^T.
        ell = args.ell;          % The vector of positive labels to use.
        C = args.C;              % Regularization parameter.
        loss = args.lossfunction;% What loss function to use.
        
        % If the slices argument is populated, we assume parallel Linear
        % SVM is being run, and slices contains the data slicing to use.
        if isfield(args, 'slices')
            
            % Get the slices, the number of slices, init the list of index
            % ranges we slice over, and the slices of D.
            slices = args.slices;
            slicelength = length(slices);
            sliceranges = cell(slicelength, 1);
            Di = cell(slicelength, 1);
            runningsum = 0;                     % Used to set up ranges.
            
            % Purpose: Populate the slice ranges and slices of D.
            for i = 1:slicelength
                sliceranges{i} = [runningsum+1, runningsum+slices(i)];
                Di{i} = D(runningsum+1:runningsum+slices(i), :);
                runningsum = runningsum + slices(i);
            end
            
            % Return the proximal operators.
            minx = 0;
            minz = @zminParallelLinearSVM;
        else
            Dplus = args.Dplus;  % Cached Pseudoinverse of D. Can be 
                                 % created using the pinv function.
            % Return the appropriate proximal operators.
            minx = @xminLinearSVM;
            minz = @zminLinearSVM;
        end
    % ---------------------------------------------------------------------
    % LASSO Problem (see below) -------------------------------------------
    case 'lasso'
        % Returns the proximal operators for solving the LASSO problem 
        % using ADMM. LASSO minimizes for x the objective function:
        %    obj(x) = 1/2*||D*x - s||_2^2 + lambda*||x||_1, 
        % where D is a data matrix and s is a signal column vector of 
        % appropriate length. Thus, x is a column vector of the same 
        % length. The parameter lambda here is the regularization
        % parameter.
        % 
        % The ADMM Augmented Lagrangian here is:
        %    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + 
        %                   constant(u),
        % where f(x) is the first term in the objective, and g(z) the 
        % second, with z = x. The rho in L_rho is the dual step size 
        % parameter. ADMM solves the equivalent problem of minimizing: 
        % f(x) + g(z), subject to x - z = 0.
        %
        % To solve this problem, we need to supply ADMM with proximal
        % operators for f and g. We can determine these as follows:
        %
        % For f, we proceed by standard minimization and take the gradient
        % of the Augmented Lagrangian, setting it equal to 0 and solving
        % for the minimizing x:
        %    grad_x(L_rho(x,z,u)) = grad_x(f(x)) + rho*(x - z + u)
        %                         = D^T*(D*x - s) + rho*(x - z + u) := 0
        % This implies:
        %    D^T*D*x + rho*x = (D^T*D + rho*I)*x = D^T*s + rho*(z - u)
        % Thus, the minimizing x is:
        %    x := (D^T*D + rho*I)^{-1}[D^T*s + rho*(z - u)]             (1)
        % To efficiently compute this, we precompute and save D^T*D, D^T*s
        % and the Cholesky decomposition L*U = R*R' = D^T*D + rho*I. The 
        % addition of rho*I can be efficiently computed by vectorizing the
        % addition of rho along the diagonal of the cached D^T*D. The
        % linear system in (1), can be expressed as:
        %    (D^T*D + rho*I)*x = D^T*s + rho*(z - u)
        % By substitution:
        %    L*U*x = D^T*s + rho*(z - u)                                (2)
        % Let U*x = y. Then, we see that solving (2) for x is equivalent to
        % solving the linear system L*y = D^T*s + rho*(z - u) for y, which
        % can be done efficiently using Matlab's backslash operator, and
        % then solving the system U*x = y for x, by the same means. Note
        % that there are no dimension restrictions on D; D^T*D will always
        % be square. If D is a very tall matrix, D^T*D will be a very small
        % square matrix. If D is very fat, D*D^T will be a very small
        % square matrix. To take advantage of this fact, we can use D*D^T
        % by utilizing the Matrix Inversion theorem, and solving the system
        % by:
        %    x := (I - D^T*(D*D^T)^{-1}*D)*(z - u) + D^T*(D*D^T)^{-1}*s
        % Inverting D*D^T can be done efficiently in the same manner as
        % above.
        % 
        % For g, we simply minimize this using soft-thresholding on v = u +
        % z and t = lambda/rho.
        %
        % As an alternative, we can solve this problem in a parallel
        % manner, by performing consensus LASSO. In consensus LASSO, we
        % split up our problem matrix D and signal vector s into row
        % segments (called slices), that can be of varying size. Then, we
        % simply perform the above algorithm (normal LASSO) on the slice of
        % data. If we do this over all the slices (in parallel), then we
        % can average the solutions on each slice to give the consensus
        % solution. In general, this is called Group LASSO. Thus, the
        % x-minimization step is done in parallel, with each solution
        % performing LASSO on their data subset. For each slice i, the x_i
        % that is computed, along with the prior u_i is summed and averaged
        % along all slices, to give the z-update. Each u_i is subsequently
        % updated using this z and individual x_i and prior u_i.
        
        rhoprev = args.rho;
        
        if args.parallel
            D = args.D;
            s = args.s;
            lambda = args.lambda;
            slices = args.slices;
            
            [m, n] = size(D);
            z = zeros(n, 1);
            slicenum = length(slices);
            runningsum = 0;
            sliceranges = cell(slicenum, 1);
            Di = cell(slicenum, 1);
            Dti = cell(slicenum, 1);
            Dtsi = cell(slicenum, 1);
            xi = cell(slicenum, 1);
            ui = cell(slicenum, 1);
            xave = zeros(n, 1);
            xaveprev = zeros(n, 1);

            for i = 1:slicenum
                sliceranges{i} = ...
                    [runningsum+1, runningsum+slices(i)];
                Di{i} = D(runningsum+1:runningsum+slices(i), :);
                Dti{i} = Di{i}';
                Dtsi{i} = ...
                    Dti{i}*s(runningsum+1:runningsum+slices(i));
                runningsum = runningsum + slices(i);
                xi{i} = zeros(n, 1);
                ui{i} = zeros(n, 1);
            end

            DtDi = cell(slicenum, 1);
            Pi = cell(slicenum, 1);
            Li = cell(slicenum, 1);
            Lti = cell(slicenum, 1);

            parfor i = 1:slicenum
                [mi, ~] = size(Di{i});
                
                % Get an LU decomposition for the x-minimization step.
                if(mi >= n)                 % D is square or tall.
                    DtDi{i} = Dti{i}*Di{i};
                else                        % D is short and fat.
                    DtDi{i} = Di{i}*Dti{i};
                end
                
                Pi{i} = DtDi{i};
                Pi{i}(1:n+1:end) = DtDi{i}(1:n+1:end) + rhoprev;
                    
                % Get lower and then upper triangular Cholesky 
                % decomposition.
                Li{i} = chol(Pi{i}, 'lower');
                Lti{i} = Li{i}';
            end
            
            % Return proximal operators.
            minx = @xminParallelLASSO;
            minz = @zminParallelLASSO;
            extra.altu = @altuLASSO; 
            extra.specialnorms = @lassonorms;
        else
            % Cache data needed for serial LASSO.
            D = args.D;
            Dts = args.Dts;
            lambda = args.lambda;
            L = args.L;
            U = args.U;
            m = args.m;
            n = args.n;
            
            % Return proximal operators.
            minx = @xminLASSO;
            minz = @(x, ~, u, rho) zminSoftThresholding(u + x, lambda/rho);
        end
    % ---------------------------------------------------------------------
    % Linear Program Problem (see below) ----------------------------------
    case 'linearprogram'
        % Returns the proximal operators for the Linear Program problem. 
        % Linear Programming minimizes for x the objective function:
        %    obj(x) = <b,x> = b^T*x, 
        %    subject to D*x = s, x >= 0
        % where D is a matrix and s is a column vector of appropriate 
        % length. Thus, x and b are column vectors of the same length. 
        % Vector b represents a vector of coefficients in the linear 
        % program. Note that this formulation is known as the standard form
        % for a linear program. One can use any conic constraint on x, not 
        % just x >= 0. In this case, one will have to provide the 
        % appropriate proximal function for g to minimize this in the 
        % options struct (args.altproxg).
        % 
        % The ADMM Augmented Lagrangian here is:
        %    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + 
        %                   constant(u),
        % where f(x) = b^T*x such that x is in the set {x: D*x = s} and 
        % g(z) is the indicator function that z is not in the non-negative 
        % orthant (R_+^n). The rho in L_rho is the dual step size 
        % parameter. ADMM solves the equivalent problem of minimizing: 
        %    f(x) + g(z), subject to x - z = 0.
        %
        % ADMM requires proximal operators for minimizing L_rho(x,z,u) for 
        % x and z, separately - i.e., proximal operators for functions f 
        % and g. The proximal operators are obtained as follows:
        % 
        % For the proximal function for f, we note that it is equal to the 
        % gradient of L_rho(x,z,u) set to zero:
        %    grad_x(L_rho(x,z,u)) := 0 
        %       <--> grad_x(b^T*x) + rho*(x - z + u) = 0
        %       <--> b + rho*(x - z + u) = 0
        %       <--> rho*x + [b - rho*(z - u)] = 0                      (1)
        % We note that this can be solved trivially as a linear system. 
        % However, the solution from (1) is not dependent on D nor s. We 
        % could stick this condition into the z-update, but there's a 
        % another way. Furthermore, as D is not square, we can't assume 
        % that we can directly solve the D and s system using D's inverse. 
        % Instead, we form a larger, square system with the properties we 
        % want and solve that. Namely, we want a square system such that:
        %    [ rho*I, F ]   [ x ]   [ b - rho*(z - u) ]   [0]
        %    [   G  , H ] * [ y ] + [        q        ] = [0]           (2)
        % where, for D being an m by n matrix, F is n by m, G is m by n, H 
        % is n by m, x and b are vectors of length n, and y and q are 
        % vectors of length m. We choose y to satisfy our domain constraint
        % on function f:
        %    D*y = s <--> D*y - s = 0                                   (3)
        % This coincides nicely with the right hand side of (1) and (2). 
        % Thus, we choose G = D. We also need the contribution of F, H and 
        % q in the right hand side of (2) to be 0. For H, that can be 
        % trivial; set H = 0. For F, however, not so much. We notice that 
        % by using (3), we naturally make q = -s. Then, we don't want F's 
        % contribution to change the corresponding 0 vector on the left 
        % hand side. So, we make a non-trivial requirement:
        %    F*y = 0 <--> D^T*y = 0                                     (4)
        % With no data for F, we arbitrarily choose F = D^T, for 
        % convenience in dimension and the ready availability of the 
        % transpose of D. So, pooling together (1 - 4) gives us a system to
        % solve:
        %    [ rho*I, D^T ]   [ x ]   [ b - rho*(z - u) ]    [0]
        %    [   D  ,  0  ] * [ y ] + [       -s        ] =  [0]        (5)
        % The solution being (now that the problem is square and has an 
        % inverse:
        %        [ x ]          ([ rho*I, D^T ])   [ rho*(z - u) - b ]
        %    v = [ y ] = inverse([   D  ,  0  ]) * [        s        ]
        % Our minimized x in the proximal function is now the x-part in 
        % solution vector v.
        %
        % For the proximal function for g, our proximal operator is simple.
        % To project a given vector v = x + u into the non-negative 
        % orthant, we simply take the positive parts of v and set the rest 
        % to 0; this is our minimized vector z.
        
        D = args.D;             % The given D matrix in D*x = s.
        Dt = args.Dt;           % Cached transpose of D.
        In = args.In;           % Cached identity matrix of size n.
        zero = args.zero;       % Cached zero matrix in the system above.
        b = args.b;             % The given coefficient vector b.
        s = args.s;             % The given vector s in D*x = s.
        n = args.n;             % Size of n for D an m by n matrix.
        
        % Return the appropriate proximal operators for minimizing x and z.
        minx = @xminLinearProgram;
        minz = @zminLinearProgram;
    % ---------------------------------------------------------------------
    % Quadratic Program Problem (see below) -------------------------------
    case 'quadraticprogram'
        % Returns the proximal operators for the Quadratic Program problem. 
        % Quadratic Programming minimizes for x the objective function:
        %    obj(x) = 1/2*<x,P*x> + <q,x> + r = 1/2*x^T*P*x + q^T*x + r, 
        %    subject to D*x = s, x >= 0
        % where D is a matrix and s is a column vector of appropriate 
        % length. Thus, x and b are column vectors of the same length. 
        % Matrix P and vector q represent coefficients in the strictly
        % quadratic and strictly linear parts of the program. The value r
        % is a constant. We assume P is a square, nonnegative matrix. Note
        % that this formulation is known as the standard form for a 
        % quadratic program. One can use any conic constraint on x, not 
        % just x >= 0. In this case, one will have to provide the 
        % appropriate proximal function for g to minimize this in the 
        % options struct (args.altproxg).
        % 
        % The ADMM Augmented Lagrangian here is:
        %    L_rho(x,z,u) = f(x) + g(z) + (rho/2)(||x - z + u||_2)^2 + 
        %                   constant(u),
        % where f(x) = 1/2*x^T*P*x + q^T*x + r such that x is in the set 
        % {x: D*x = s} and g(z) is the indicator function that z is not in
        % the non-negative orthant (R_+^n). The rho in L_rho is the dual 
        % step size parameter. ADMM solves the equivalent problem of 
        % minimizing: 
        %    f(x) + g(z), subject to x - z = 0.
        %
        % ADMM requires proximal operators for minimizing L_rho(x,z,u) for 
        % x and z, separately - i.e., proximal operators for functions f 
        % and g. The proximal operators can be obtained in exactly the same
        % method as in a Linear Program (see above), except with the
        % inclusion of the matrix P in the solution. That is, the proximal
        % operator for f is:
        %    [ P + rho*I, D^T ]   [ x ]   [ q - rho*(z - u) ]    [0]
        %    [     D    ,  0  ] * [ y ] + [       -s        ] =  [0]
        % The solution being (now that the problem is square and has an 
        % inverse:
        %        [ x ]          ([ P + rho*I, D^T ])   [ rho*(z - u) - q ]
        %    v = [ y ] = inverse([     D    ,  0  ]) * [        s        ]
        % Our minimized x in the proximal function is now the x-part in 
        % solution vector v.
        % 
        % For the proximal function for g, our proximal operator is simple.
        % To project a given vector v = x + u into the non-negative 
        % orthant, we simply take the positive parts of v and set the rest 
        % to 0; this is our minimized vector z.
        %
        % Alternatively, we can simplify the constraint in our objective
        % function to be:
        %    obj(x) = 1/2*<x,P*x> + <q,x> + r = 1/2*x^T*P*x + q^T*x + r, 
        %    subject to lb <= x <= ub
        % In this case, we can take a different approach. We assume that P
        % can be Cholesky factored into P = X^T*X. Furthermore, if we
        % perturb P with rho*I, then there must exist a square matrix R
        % such that P + rho*I = R^T*R, via the same Cholesky factorization.
        % Then, we can minimize for function f by taking the gradient and
        % solving for x, when set to 0:
        %    grad_x(L_rho(x,z,u)) := 0 
        %       <--> grad_x(1/2*x^T*(X^T*X)*x + q^T*x + r) + 
        %            rho*(x - z + u) = 0
        %       <--> grad_x(1/2*||X*x||^2) + q + rho*(x - z + u) = 0
        %       <--> X^T*X*x + rho*x + [q - rho*(z - u)] = 0
        %       <--> (X^T*X + rho*I)*x = rho*(z - u) - q
        %       <--> (R^T*R)*x = rho*(z - u) - q                        (1)
        %       <--> R^T*(R*x) = rho*(z - u) - q
        %       <--> R^T*y = rho*(z - u) - q                            (2)
        % Assuming we know R, (1) can be efficiently solved by solving (2)
        % for y, and then solving R*x = y for x. For the proximal operator
        % for g, where g in this situation indicates whether z is NOT in
        % the set {z: lb <= z <= ub}, thus the minimizing z for function g
        % is simply a min-max between z and the bounds lb and ub:
        %    z_i := min(ub_i, max(lb_i, v_i))
        % where v = x + u.
        % 
        % To specify which strategy to use, set args.constraint equal to
        % either the string 'standard' or 'bounded'. Alternatively, one can
        % provide the g proximal operator for a different conic constraint,
        % in args.altproxg. The constraint defaults to standard.
        
        % Specifies which constraint we use.
        constraint = args.constraint;
        
        P = args.P;                 % The given coefficient matrix P.
        q = args.q;                 % The given coefficient vector b.
        rho = args.rho;             % Dual step length rho.
        
        % Assign parameters based of the constraint type.
        if strcmp(constraint, 'bounded')
            lb = args.lb;           % The lower bounding vector.
            ub = args.ub;           % The upper bounding vector.
            rhoprev = args.rho;     % Last rho used. If current one differs
                                    % then we reconstruct R.
            n = args.n;             % Column size of P.
            Pnew = P;
            
            % Save the perturbed P and find its R^T*R factorization.
            Pnew(1:n+1:end) = P(1:n+1:end) + rho;
            R = chol(Pnew);
        else
            D = args.D;             % The given D matrix in D*x = s.
            Dt = args.Dt;           % Cached transpose of D.
            In = args.In;           % Cached identity matrix of size n.
            zero = args.zero;       % Cached zero matrix in the system.
            s = args.s;             % The given vector s in D*x = s.
            n = args.n;             % Size of n for D an m by n matrix.
            rhoprev = args.rho;     % Last rho used. If current one differs
                                    % then we reconstruct Pnew = P + rho*I.
            Pnew = P + rho*In;      % Caches the sum P + rho*I.
        end
        
        % Return the appropriate proximal operators for minimizing x and z.
        if strcmp(constraint, 'bounded')
            minx = @xminQuadraticProgramBounded;
            minz = @zminQuadraticProgramBounded;
        else
            minx = @xminQuadraticProgramStandard;
            minz = @zminQuadraticProgramStandard;
        end
        
        % If the user has a different bound for g, return that.
        if isfield(args, 'altproxg')
            minz = args.altproxg;
        end
    % ---------------------------------------------------------------------
    % Sparse Inverse Covariance Selection (see below) ---------------------
    case 'covarianceselection'
        % Returns the proximal operators for the Sparse Inverse Covariance
        % Selection problem. Sparse Inverse Covariance Selection minimizes
        % for x the objective function:
        %    obj(X) = trace(S*X) - log(det(X)) + lambda*||X||_1, 
        % where S = covariance(D) is a square matrix of the covariance of
        % D. Thus, X is a square matrix as well. The parameter lambda here
        % is the l_1 regularization parameter.
        %
        % In more detail, suppose we have a dataset D of samples from a
        % Gaussian distribution in R^n, with zero mean. Let a single row of
        % D, of length n, be a single sample. Then, for each row i,
        % D_i ~ Normal(0, Sigma), i = 1, ..., N, where Sigma is a matrix of
        % covariances. Suppose we wish to estimate Sigma under the
        % assumption that its inverse is sparse. Note that any entry in the
        % inverse of Sigma is 0 if and only if the corresponding components
        % of the random variable are conditionally independent, given the
        % other variables. Because of this, this problem is similar to the
        % structure learning problem of estimating the topology of the
        % undirected graph of the Gaussian. Determining the sparsity
        % pattern of the inverse of Sigma is known as the Covariance
        % Selection problem. Thus, given the sparsity, and an empirical
        % covariance of a dataset D, called S, we can formulate this
        % problem as minimizing the loss function:
        %    l(X) = trace(S*X) + log(det(X))
        % The l_1 regularization version of this problem is the obj
        % function above. Thus, this problem is referred to as Sparse
        % Inverse Covariance Selection.
        % 
        % The ADMM Augmented Lagrangian here is:
        %    L_rho(X,Z,U) = f(X) + g(Z) + (rho/2)(||X - Z + U||_2)^2 +
        %                   constant(U),
        % where f(X) = l(X) and g(Z) = lambda*||Z||_1 is the regularization
        % term. The rho in L_rho is the dual step size parameter. ADMM
        % solves the problem of minimizing: f(X) + g(Z),
        % subject to X - Z = 0.
        %
        % ADMM requires proximal operators for minimizing L_rho(X,Z,U) for
        % X and Z, separately - i.e., proximal operators for functions f
        % and g. The proximal operators are obtained as follows:
        % 
        % For the f minimization, observe that the gradient of the
        % Augmented Lagrangian requires:
        %    grad_X(L_rho(X,Z,U)) = grad_X(f(X)) + rho*(X - Z + U)
        %                         = S - 1/X + rho*(X - Z + U) := 0
        % Thus, we have the condition:
        %    rho*X - 1/X := rho*(Z - U) - S                             (1)
        % To satisfy (1), we must construct such a matrix X from Z, U, S
        % and rho only (the only prior information we have in ADMM). We
        % also have an implicit condition that X is positive semi-definite.
        % To this, we begin by taking the orthogonal eigenvalue
        % decomposition of the right-hand side of (1):
        %    rho*(Z - U) - S = Q*E*Q^T,
        % where Q contains the eigenvectors and E is a diagonal vector of
        % corresponding eigenvalues. Note that Q^T*Q = Q*Q^T = I. Thus,
        % multiplying the left-hand side of (1) by Q^T on the left and Q on
        % the right gives:
        %    Q^T*(rho*X - 1/X)*Q = rho*(Q^T*X*Q) - 1/(Q^T*X*Q)
        % and the right-hand side of (1) thus becomes:
        %    Q^T*(rho*(Z - U) - S)*Q = Q^T*(Q*E*Q^T)*Q
        %                            = (Q^T*Q)*E*(Q^T*Q)
        %                            = E
        % So, we want a matrix V = Q^T*X*Q, such that: rho*V - 1/V = E.
        % This, of course, must have a diagonal solution, as E is diagonal.
        % Using the Quadratic Formula, each i'th diagonal entry must
        % satisfy:
        %    V_{i,i} = 1/(2*rho)*[lambda_i + sqrt(lambda_i^2 + 4*rho)]
        % where lambda_i is the eigenvalue at E_{i,i}. As rho is positive,
        % these entries are also always positive. Thus, X = Q*V*Q^T
        % satisfies (1) and is our optimal X.
        %
        % For function g, the Z-update is simply soft-thresholding
        % performed on matrix X + U instead of a vector, with lambda/rho as
        % the threshold.
        
        % Populate constants for proximal operators.
        S = args.S;             % Covariance of matrix D.
        lambda = args.lambda;   % The coefficient of regularization term.
        
        % Return the proximal operators.
        minx = @xminCovarianceSelection;
        minz = @(x, ~, u, rho) zminSoftThresholding(u + x, lambda/rho);
    % ---------------------------------------------------------------------
    % Least Absolute Deviations (LAD) -------------------------------------
    case 'lad'
        % Returns the proximal operators for the Least Absolute Deviations
        % problem. Least Absolute Deviations minimizes for x the objective
        % function:
        %    obj(x) = ||D*x - s||_1, 
        % where D is a data matrix and s is a signal vector. We can
        % rephrase this in ADMM form as the following objective function to
        % minimize:
        %    obj(x,z) = ||z||_1,
        %    subject to D*x - z = s
        % Here, f(x) = 0, and g(z) = ||z||_1. The ADMM Augmented Lagrangian
        % for this problem is:
        %    L_rho(x,z,u) = g(z) + (rho/2)(||D*x - z - s + u||_2)^2 +
        %                   constant(u),
        % The rho in L_rho is the dual step size parameter. ADMM solves the
        % problem of minimizing function obj(x,z) under the constraint.
        %
        % ADMM requires proximal operators for minimizing L_rho(x,z,u) for
        % x and z, separately - i.e., proximal operators for functions f
        % and g. The proximal operators are obtained as follows:
        % 
        % For function f, we minimize by taking the gradient for x and
        % setting it equal to 0, then solving for x:
        %    grad_x(L_rho(x,z,u)) = D^T*(D*x - z - s + u) := 0
        % This implies that solving the following system for x gives our
        % minimizing x:
        %    D^T*D*x = D^T*(z + s - u)                                  (1)
        % To solve this efficiently, we do not compute (D^T*D)^{-1}
        % directly, but instead find its Cholesky decomposition,
        % D^T*D = R*R^T, where R and R^T are lower and upper triangular,
        % respectively. Then, (1) can be solved by first solving the system
        % R*y = D^T*(z + s - u) for y, and then solving the system R^Tx = y
        % for x. We cache the factors R and R^T and use system solving
        % backslash operator A \ b to solve a linear system A*x = b
        % efficiently.
        %
        % For function g, we minimize by soft-thresholding, as this is a
        % l_1 regularization problem. The soft-thresholding parameters here
        % are v = D*x - s + u and t = 1/rho.
        
        % Obtain arguments from the args struct.
        R = args.R;
        D = args.D;
        s = args.s;
        
        % Cache the transposes of these matrices.
        Rt = R';
        Dt = D';
        
        % Return the proximal operator for f for this problem.
        minx = @xminLAD;
        
        % Return the proximal operator for g for this problem, depending on
        % whether or not relaxation is being used.
        if (isfield(args, 'userelax') && args.userelax)
            minz = @(x, ~, u, rho) zminSoftThresholding(x + u - s, 1/rho);
        else
            minz = @(x, ~, u, rho)zminSoftThresholding(D*x + u - s, 1/rho);
        end
    % ---------------------------------------------------------------------
    % HUBER FITTING (see below) -------------------------------------------
    case 'huberfit'
        % Returns the proximal operator for the Huber Fitting problem.
        % Huber Fitting fits data according to a curve defined by the Huber
        % function. In this context, the Huber function is defined as:
        %    huber(a) = { a^2/2         |a| <= 1 }
        %               { |a| - 1/2     |a| >  1 }
        % For vector arguments, this function is applied over all
        % components and the results aggregated. The Huber Fitting problem
        % is defined as minimizing for x the objective function:
        %    obj(x) = 1/2*sum(huber(D*x - s)), 
        % where D is a data matrix and s is a signal vector. We can
        % rephrase this in ADMM form as the following objective function to
        % minimize:
        %    obj(x,z) = 1/2sum(huber(z)),
        %    subject to D*x - z = s
        % Here, f(x) = 0, and g(z) = obj(z). The ADMM Augmented Lagrangian
        % for this problem is:
        %    L_rho(x,z,u) = g(z) + (rho/2)(||D*x - z - s + u||_2)^2 + 
        %                   constant(u),
        % The rho in L_rho is the dual step size parameter. ADMM solves the
        % problem of minimizing function obj(x,z) under the constraint.
        %
        % ADMM requires proximal operators for minimizing L_rho(x,z,u) for
        % x and z, separately - i.e., proximal operators for functions f
        % and g. The proximal operators are obtained as follows:
        % 
        % For function f, we minimize by taking the gradient for x and
        % setting it equal to 0, then solving for x:
        %    grad_x(L_rho(x,z,u)) = D^T*(D*x - z - s + u) := 0
        % This implies that solving the following system for x gives our
        % minimizing x:
        %    D^T*D*x = D^T*(z + s - u)                                  (1)
        % To solve this efficiently, we do not compute (D^T*D)^{-1}
        % directly, but instead find its Cholesky decomposition,
        % D^T*D = R*R^T, where R and R^T are lower and upper triangular,
        % respectively. Then, (1) can be solved by first solving the system
        % R*y = D^T*(z + s - u) for y, and then solving the system R^Tx = y
        % for x. We cache the factors R and R^T and use system solving
        % backslash operator A \ b to solve a linear system A*x = b
        % efficiently.
        %
        % For function g, we simply use the proximal operator for the Huber
        % function and minimize that. To be more explicit, we minimize via
        % the gradient as for f:
        %    grad_z(L_rho(x,z,u)) = sum(grad_z(huber(z))) - 
        %                           rho*(D*x - z - s + u)
        % In the case of a component of z, z_i being less than or equal to
        % 1, the gradient there is z_i. In the other cases, it is either -1
        % if x_i was negative, or 1 if it was positive. In the former case,
        % solving the gradient at component z_i for 0 is:
        %    z_i - rho*([D*x]_i - z_i - s_i + u_i) := 0
        % Which implies (1 + rho)*z_i = rho*([D*x]_i - s_i + u_i), or that:
        %    z_i = rho/(1 + rho)*([D*x]_i - s_i + u_i)                  (1)
        % For the latter case, the Huber gradient is constant 
        % z_i/|z_i| = +/-1, so:
        %    +/-1 - rho*([D*x]_i - z_i - s_i + u_i) := 0
        % This can be written as:
        %    z_i = ([D*x]_i - s_i + u_i) + (+/-1)/rho
        % We see that this can be split up into:
        %   z_i = rho/(1 + rho)*([D*x]_i - s_i + u_i) + ...             (2)
        %         1/(1 + rho)*[[D*x]_i - s_i + u_i) + (+/-1)*(1 + rho)/rho]
        % Combining the two cases for the Huber function's gradient, we see
        % that (1) and (2) differ by the second term in (2), and so in case
        % |z_i| <= 1 we set the second term to 0 and in case |z_i| > 1, we
        % keep the second term. For the bracketed expression in (2), this
        % corresponds to the cases of soft thresholding over a step of
        % t = (1 + rho)/rho = 1 + 1/rho. So, we have that the minimizing z
        % can be written in a single statement as:
        %    z = rho/(1 + rho)*(D*x - s + u) + ...
        %        1/(1 + rho)*S(D*x - s + u,1 + 1/rho)
        % where function S(v,t) is the soft-thresholding operator for
        % vector v and parameter t. Setting v = D*x - s + u and factoring
        % out 1/(1 + rho), we have an efficient expression for the proximal
        % operator as:
        %    z = 1/(1 + rho)*[rho*v + S(v, 1 + 1/rho)]
        
        % Obtain arguments from the args struct.
        R = args.R;
        D = args.D;
        s = args.s;
        
        % Cache the transposes of these matrices.
        Rt = R';
        Dt = D';
        
        % Return the proximal operator for f for this problem. In this
        % case, we can use the same proximal operator as for the LAD
        % problem, as f(x) = 0 here as well.
        minx = @xminLAD;
        
        % Return the proximal operator for g for this problem, depending on
        % whether or not relaxation is being used.
        if (isfield(args, 'userelax') && args.userelax)
            minz = @(Dxhat, ~, u, rho) ...
                zminHuberSoftThresholding(Dxhat, 0, u, rho); 
        else
            minz = @(x, ~, u, rho) ...
                zminHuberSoftThresholding(D*x, 0, u, rho);
        end
    % ---------------------------------------------------------------------
    % Default to not doing anything but report an error.
    otherwise
        error('Invalid input for problem - given string is not a solver!');
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% The general proximal operator for l1 regularization minimization. Note 
% that this is minimizable via soft-thresholding, a proximal mapping 
% technique:
%    z = min_z(||z - v||_2^2 + lambda*||v||_1) 
%      = sign(z)*(|z| - t)_+ (non positive parts assigned 0)
% This function simply evaluates this for v, and appropriate threshold t 
% (genarally equal to lambda/rho, or 1/rho, etc.).

function [minz] = zminSoftThresholding(v, t)
% Same INPUTS and OUTPUTS as in the model case for z (above).

    % Evaluate the soft thresholding as described in the description.
    minz = sign(v).*subplus(abs(v) - t);
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Evaluates the model problem's (see main function above) proximal operator
% for corresponding function f. The proximal operator is the result of
% taking an x derivative of the model's L_rho and solving it for 0, i.e.:
%    P^T*(P*x - r) + rho*(x - z + u) = 0
% solved for x. This involves solving a simple linear system.

function [minx] = xminModel(~, z, u, rho)
% INPUTS ------------------------------------------------------------------
% x:    The x input corresponding to the function f(x) in the model
%       problem. See main function above for more details. This input is
%       excluded as it is not necessary in the minimization, but is part of
%       the function call and must be included anyway.
% z:    The z input corresponding to the function g(z) in the model
%       problem. See main function above for more details.
% u:    The Lagrange Multiplier variable in ADMM's Augmented Lagrangian
%       L_rho(x, z, u). See main function above for more details.
% rho:  The step size parameter for ADMM.
%
% OUTPUTS -----------------------------------------------------------------
% minx: The minimal x as described in the description.
% -------------------------------------------------------------------------

    if (rho ~= rhoprev)
        % Efficiently add rho to the diagonal entries of P^T*P.
        PtPnew(1:n+1:end) = PtP(1:n+1:end) + rho;
    end
    
    % Solve the linear system from the description for minimal x.
    minx = PtPnew \ (Ptr + rho*(z - u));
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Evaluates the model problem's (see main function above) proximal operator
% for corresponding function g. The proximal operator is the result of
% taking a z derivative of the model's L_rho and solving it for 0, i.e.:
%    Q^T*(Q*z - s) - rho*(x - z + u) = 0
% solved for z. This involves solving a simple linear system.

function [minz] = zminModel(x, ~, u, rho)
% INPUTS ------------------------------------------------------------------
% x:    The x input corresponding to the function f(x) in the model
%       problem. See main function above for more details. 
% z:    The z input corresponding to the function g(z) in the model
%       problem. See main function above for more details. This input is
%       excluded as it is not necessary in the minimization, but is part of
%       the function call and must be included anyway.
% u:    The Lagrange Multiplier variable in ADMM's Augmented Lagrangian
%       L_rho(x, z, u). See main function above for more details.
% rho:  The step size parameter for ADMM.
%
% OUTPUTS -----------------------------------------------------------------
% minz: The minimal z as described in the description.
% -------------------------------------------------------------------------
    
    if (rho ~= rhoprev)
        % Efficiently add rho to the diagonals of Q^T*Q.
        QtQnew(1:n+1:end) = QtQ(1:n+1:end) + rho;
    end
    
    % Solve the linear system from the description for minimal z.
    minz = QtQnew \ (Qts + rho*(x + u));
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Evaluates the Basis Pursuit problem's (see main function above) proximal
% operator for corresponding function f. As f indicates if x is not in the
% set {x: D*x = s}, its minimum occurs when x IS in the set. We can
% guarantee to find such an x by projecting a vector v = z - u onto this
% set. This is done using the cached P and q, by computing the
% Psuedoinverse of D times v, P*(z - u) + q.

function [minx] = xminBasisPursuit(~, z, u, ~) 
% Same INPUTS and OUTPUTS as in the model case for x (above).

    % Computes the minimum as described in the description.
    minx = P*(z - u) + q;
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Returns the minimized x via the proximal operator for f for the Total
% Variation Minimization problem. We simply solve the system described in
% the TVM section of getproxops to find this value.

function [minx] = xminTotalVariation(~, z, u, rho)
% Same INPUTS and OUTPUTS as in the model case for x (above).

    minx = (Id + rho*DtD) \ (s + rho*Dt*(z - u));
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Returns minimizing x via the proximal operator for f for the Linear SVM
% problem. Note that f's formulation is minimizable by transpose reduction,
% forming the pseudoinverse of data matrix D, known as D^+, and multiplying
% it by the given vector v. This function simply evaluates this for 
% v = z - u. 

function [minx] = xminLinearSVM(~, z, u, ~)
% Same inputs and outputs as for the model problem's xmin case above. The
% input rho is not used here.
    
    % Multiplies z - u by the cached D^+ matrix.
    minx = Dplus*(z - u);
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Returns minimizing z via the proximal operator for g for the Linear SVM
% problem. Note that g's formulation is minimizable by simply taking the
% gradient of the Augmented Lagrangian, setting it equal to 0, and solving
% the system. The hard part is minimizing the loss function; generally this
% is the hinge loss function but you can also use the 0-1 loss function
% thanks to the transpose reduction and unwrapped ADMM. Read the linearsvm
% entry in the main getproxops function above to see how to minimize each.

function [minz] = zminLinearSVM(x, ~, u, rho)
% Same inputs and outputs as for the model problem's zmin case above.

    % Cache some repeated computations.
    Dx = D*x;                                   % Save this product.
    Dxplusu = Dx + u;                           % Save this sum.
    v = ell.*Dxplusu;                           % Save label product.
    
    % The z-minimization step, based on whether to use Hinge or 0-1 loss
    % function.
    if ~strcmp(loss, '01')                      % Minimize the hinge loss.
        % Proximal operator with minimized hinge loss as second term.
        minz = Dxplusu + ell.*max(min(1 - v, C/rho), 0);
    else                                        % Minimize the 0-1 loss.
        % Proximal operator with minimized 0-1 loss (function min01 is the
        % function below this one).
        minz = ell.*minz01(v, rho/C);
    end
    
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Returns minimizing parallel segment z_i via the proximal operator for g 
% for the Linear SVM problem. Note that g's formulation is minimizable by
% simply taking the gradient of the Augmented Lagrangian, setting it equal
% to 0, and solving the system. The hard part is minimizing the loss 
% function; generally this is the hinge loss function but you can also use
% the 0-1 loss function thanks to the transpose reduction and unwrapped 
% ADMM. Read the linearsvm entry in the main getproxops function above to 
% see how to minimize each.

function [minz] = zminParallelLinearSVM(x, ~, u, rho, i)
% Same inputs and outputs as for the model problem's zmin case above. The
% only exception is additional input i for the ith slice in parallel
% computation of minimizing z for proximal operator g.

    slice = sliceranges{i};
    
    % Cache some repeated computations.
    Dx = Di{i}*x;                               % Save this product.
    Dxplusu = Dx + u(slice(1):slice(2));        % Save this sum.
    v = ell(slice(1):slice(2)).*Dxplusu;        % Save label product.
    
    % The z-minimization step, based on whether to use Hinge or 0-1 loss
    % function.
    if ~strcmp(loss, '01')                      % Minimize the hinge loss.
        % Proximal operator with minimized hinge loss as second term.
        minz = Dxplusu + ell(slice(1):slice(2)).*max(min(1 - v, C/rho), 0);
    else                                        % Minimize the 0-1 loss.
        % Proximal operator with minimized 0-1 loss (function min01 is the
        % function below this one).
        minz = ell(slice(1):slice(2)).*minz01(v, rho/C);
    end
    
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Returns the minimized value of the 0-1 loss function for the proximal
% operator for g in the Linear SVM problem (the proximal operator being the
% function immediately above this one). We wish to get 
%    argmin_y(1/(2*t)*||y - s||_2^2 + zo(y)),
% where zo is the 0-1 loss function. See the main function getproxops
% above, and read the linearsvm section to see how this is minimized.

function y = minz01(s, t)
% INPUTS: 
% s     Vector to use in proximal operator.
% t     Proximal parameter, rho/C.
% -------------------------------------------------------------------------
% OUTPUTS:
% y     Result of proximal operator.
% -------------------------------------------------------------------------

% Initialization; a value of 1 in y indicates non-classification as the 0-1
% loss function is the indicator function of whether or not a component was
% NOT classified correctly. Thus, the minimum is correct classification
% (returned value of 0).
y = ones(length(s), 1);

% Compute indices in s where the labelling components would be indicated as
% 0 and isolate those indices from s in y.
inds = (s >= 1) | (s < (1 - sqrt(2/t)));

% Now we set those locations in y to be s's corresponding value.
y(inds) = s(inds);

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% The x-minimization step for LASSO. Solves the system in described in the
% 'lasso' section of getproxops's switch statement using factorized upper
% and lower matrices.

function [minx] = xminLASSO(~, z, u, rho)
% Same inputs and outputs as the model problem for x above.
    
    y = rho*(z - u) + Dts;     % For ease of writing.

    % Solve the system efficiently based on whether A was short or tall.
    if (m >= n)                  % A is square or tall.
        % Solve system via LU-decomposition efficiently.
        minx = U \ (L \ y);
    else                        % A is short and fat.
        % Solve via LU-decomposition with roles of A and A transpose 
        % swapped. We get a solution of length n, as desired.
        minx = 1/rho*y - 1/rho^2*(D'*(U \ (L \ (D*y))));
    end
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% The x-minimization step for parallel LASSO. Runs steps in parallel and
% returns updated x.

function [minx] = xminParallelLASSO(~, ~, ~, rho)
% Same inputs and outputs as the model problem for x above.

    minx = zeros(n, 1);
    
    if rho ~= rhoprev
        newrho = 1;
    else
        newrho = 0;
    end
    
    parfor k = 1:slicenum
        % If a different rho is used, reconstruct Pnew = P + rho*I and cache
        % the rho used to do that.
        if newrho
            Pi{k}(1:n+1:end) = DtDi{k}(1:n+1:end) + rho;
            rhoprev = rho;

            % Get lower and then upper triangular Cholesky decomposition.
            Li{k} = chol(Pi{k}, 'lower');
            Lti{k} = Li{k}';
        end

        yi = rho*(z - ui{k}) + Dtsi{k};     % For ease of writing.

        [mi, ~] = size(Di{k});

        % Solve the system efficiently based on whether D was short or tall.
        if (mi >= n)                       % D_i is square or tall.
            % Solve system via LU-decomposition efficiently.
            xi{k} = Lti{k} \ (Li{k} \ yi);
        else                                % D_i is short and fat.
            % Solve via LU-decomposition with roles of D_i and D_i transpose 
            % swapped. We get a solution of length ni, as desired.
            xi{k} = yi/rho - 1/rho^2*(Dti{k}*(Lti{k} \ (Li{k} \ (Di{k}*yi))));
        end
    end
    
    parfor k = 1:slicenum
        minx = minx + xi{k};
    end
    
    minx = minx./slicenum;
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Returns the proximal operator for z in the Parallel LASSO problem; i.e.,
% performs soft-thresholding with lambda/rho on average(x) + average(u), as
% per Global Consensus over parallel solutions.

function [minz] = zminParallelLASSO(~, ~, ~, rho)
% Same inputs and outputs as the model problem above.
    
    uave = zeros(n, 1);
    minz = uave;
    xaveprev = xave;
    xave = zeros(n, 1);

    % Add up all slice components.
    parfor j = 1:slicenum
        uave = uave + ui{j};
        xave = xave + xi{j};
    end
    
    % Compute average.
    uave = uave./slicenum;
    xave = xave./slicenum;
    
    % Update z.
    v = uave + xave;
    z = sign(v).*subplus(abs(v) - lambda/(rho*slicenum));
    
    % Immediately perform u-update. Can be done in parallel as order of
    % addition does not matter.
    parfor j = 1:slicenum
        ui{j} = ui{j} + (xi{j} - z);
    end
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Alternate u-update for consensus LASSO. Aggregates all u_i into a single
% u variable and average it by dividing over the i slices. This averaged u
% value is the agreed-upon solution by the i sub-problems, as per consensus
% LASSO.

function u = altuLASSO(~, ~, ~, ~)
% Same INPUTS as any other proximal operator, as a user-defined u update 
% might need ALL this information (though in this case we need none of it).
% The OUTPUT is u, the consensus LASSO averaged u-update.
    
    u = zeros(n, 1);            % Init our u variable.
    
    % Aggregate u_i across all slices. Can be done in parallel as order of
    % addition does not matter.
    parfor j = 1:slicenum
        u = u + ui{j};
    end
    
    u = u./slicenum;            % The division step of averaging.
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Norms for Consensus LASSO.
function v = lassonorms(~, ~, ~, ~)
    v = [0 0];
    
    for j = 1:slicenum
        v(1) = v(1) + norm(xi{j} - xave, 'fro')^2;
    end
    
    v(2) = slicenum*rhoprev^2*norm(xave - xaveprev, 'fro')^2;
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Evaluates the Linear Program problem's (see main function above) proximal
% operator for corresponding function f. Note that f's formulation is
% minimizable via setting up the system described in the main function.
% This function simply solves this system and returns the appropriate first
% segment of it (first n entries).

function [minx] = xminLinearProgram(~, z, u, rho)
% Same INPUTS and OUTPUTS as in the model case for x (above).

    % Solve for vector v described in the main function. Return the first n
    % components of this vector - this is our minimized x as per the
    % proximal operator.
    xtmp = [rho*In, Dt; D, zero] \ [rho*(z - u) - b; s];
    minx = xtmp(1:n);
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Evaluates the Linear Program problem's (see main function above) proximal
% operator for corresponding function g. Note that g's formulation is
% minimizable via simply projecting the vector x + u into the non-negative
% orthant by returning the positive components and setting the rest to 0.

function [minz] = zminLinearProgram(x, ~, u, ~)
% Same INPUTS and OUTPUTS as in the model case for z (above).

    minz = pos(x + u);
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Evaluates the Quadratic Program problem's (see main function above)
% proximal operator for corresponding function f, in standard form. Note
% that f's formulation is minimizable via setting up the system described 
% in the main function (same as Linear Program with quadratic term). This
% function simply solves this system and returns the appropriate first
% segment of it (first n entries).

function [minx] = xminQuadraticProgramStandard(~, z, u, rho)
% Same INPUTS and OUTPUTS as in the model case for x (above).

    % If a different rho is used, reconstruct Pnew = P + rho*I and cache
    % the rho used to do that.
    if rho ~= rhoprev
        Pnew(1:n+1:end) = P(1:n+1:end) + rho;
        rhoprev = rho;
    end
    
    % Solve for vector v described in the main function. Return the first n
    % components of this vector - this is our minimized x as per the
    % proximal operator.
    xtmp = [Pnew, Dt; D, zero] \ [rho*(z - u) - q; s];
    minx = xtmp(1:n);
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Performs exactly the same operation as zminLinearProgram.

function [minz] = zminQuadraticProgramStandard(x, ~, u, ~)
% Same INPUTS and OUTPUTS as in the model case for z (above).

    minz = pos(x + u);
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Evaluates the Quadratic Program problem's (see main function above)
% proximal operator for corresponding function f, in bounded form. Note
% that f's formulation is minimizable via finding the Cholesky
% decomposition P + rho*I = R^T*R and using R and R^T to solve for the
% correct minimizer, as described in the 'quadraticprogram' section of the
% main getproxops function.

function [minx] = xminQuadraticProgramBounded(~, z, u, rho)
% Same INPUTS and OUTPUTS as in the model case for x (above).

    % If a different rho is used, reconstruct decomposition P + rho*I = 
    % R^T*R and cache the rho used to do that.
    if rho ~= rhoprev
        % Efficiently perturb the diagonal of P by rho, saving it in our
        % already populated Pnew variable for space/time efficiency. Then,
        % get the Cholesky decomposition of Pnew = P + rho*I.
        Pnew(1:n+1:end) = P(1:n+1:end) + rho;
        R = chol(Pnew);
        rhoprev = rho;
    end
    
    minx = R \ (R' \ (rho*(z - u) - q));
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% For the minimization for g, we need to find the minimumizing vector minz
% that belongs in the set {z: for all i, lb_i <= z_i <= ub_i}. This is
% obviously a min-max problem, where we take the maximum of the lower bound
% lb and x + u, and then take the minimum between the upper bound ub and
% the result. This is our minimizing z, minz.

function [minz] = zminQuadraticProgramBounded(x, ~, u, ~)
% Same INPUTS and OUTPUTS as in the model case for z (above).

    minz = min(ub, max(lb, x + u));
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Evaluates the Sparse Inverse Covariance Selection problem's (see main 
% function above) proximal operator for corresponding function f. Note that
% f's formulation is minimizable via the strategy described in the 
% 'covarianceselection' section of the main getproxops function, above.

function [minx] = xminCovarianceSelection(~, z, u, rho)
% Same INPUTS and OUTPUTS as in the model case for x (above).

    [Q, E] = eig(rho*(z - u) - S);
    e = diag(E);                    % Turn E into vector of eigenvalues.
    
    % Construct quadratic solution to minimization problem and return
    % minimal x under all eigenvectors/eigenvalue pairs.
    minx = Q*diag((e + sqrt(e.^2 + 4*rho))./(2*rho))*Q';
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Evaluates the Least Absolute Deviations problem's (see main function
% above) proximal operator for corresponding function f. Note that f's 
% formulation is minimizable via the strategy described in the 'lad'
% section of the main getproxops function, above. It involves using the
% Cholesky decomposition A^T*A = R*R^T to perform a double system solve 
% over R and R^T, instead of computing the inverse of A^T*A directly.

function [minx] = xminLAD(~, z, u, ~)
% Same INPUTS and OUTPUTS as in the model case for x (above).

    minx = Rt \ (R \ (Dt*(s + z - u)));
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
% 
% Performs soft-thresholding for Huber Fitting on function g. Generalized
% to allow A*x as input (relaxed or normal) as opposed to just x, where A
% is the matrix in ADMM's constraints. For Huber Fitting, A = D. See
% section 'huberfit' in main function getproxops above to see how this soft
% thresholding technique is derived for Huber Fitting.

function minz = zminHuberSoftThresholding(Ax, ~, u, rho)
% Same INPUTS and OUTPUTS as in the model case for x (above). The only
% difference in INPUT is that Ax can be simply A*x or relaxed A*x term, for
% matrix A in constraint term.

    v = Ax + u - s;         % Vector v in generalized proximal operator.
    
    % Evaluation of the proximal operator for Huber's soft-thresholding,
    % which involves a call to regular soft-thresholding.
    minz = 1/(1 + rho)*(rho*v + zminSoftThresholding(v, 1 + 1/rho));
end
% -------------------------------------------------------------------------

end
% -------------------------------------------------------------------------