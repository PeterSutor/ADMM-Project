% -------------------------------------------------------------------------
% Performs unwrapped ADMM for SVM training.
function results = linearSVM(D, ell, C, hinge, options)
% INPUTS:
% D         The data matrix to train on (matrix for x variable).
% ell       Labels for rows in D.
% C         Regularization parameter.
% hinge     If 1, use Hinge loss function, if 0 use 0-1 loss function.


if (isfield(options, 'rho'))
    rho = options.rho;
else
    rho = 1.0;
end

[m1, n1] = size(D);                             % Size of D.

Dt = D';                                        % Save transpose of D.
Id = eye(n1, n1);                                % Identity matrix.
Dplus = pinv(Id + rho*(Dt*D))*Dt*rho;            % Pseudo-inverse of D.

args.D = D;
args.Dt = Dt;
args.Id = Id;
args.Dplus = Dplus;
args.ell = ell;
args.C = C;
args.hinge = hinge;
[minx, minz] = getProxOps('LinearSVM', args);

options.A = D;
options.At = D';
options.B = -ones(m1, m1);
options.c = zeros(m1, 1);
options.rho = rho;

results = admm(minx, minz, options);

end

