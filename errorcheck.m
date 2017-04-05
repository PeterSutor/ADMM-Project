% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Performs error checks on various types of input. The argument arg is the
% input being check, str is the string representation of the variable's
% name in the context it is being used, and check is a string containing
% the type of check that we are performing. For example, if you wanted to
% check that arg is a numeric matrix, set check = 'ismatrix'. If arg is not
% a matrix, function errorcheck will return an error message referring to
% arg by the string str. If arg can be altered to make it fit the check,
% errorcheck will alter it and return it as the output. For example, if
% given a positive complex number when checking for a positive real number,
% error check will return real(arg).

function arg = errorcheck(arg, check, str, options)
% INPUTS ------------------------------------------------------------------
% arg:      The argument we are performing a check on.
% check:    A string containing the type of check to perform. See the
%           switch statement below to see what kind of check you can
%           perform.
% str:      A string containing the name of the argument arg in the context
%           it is being used. This is used to refer to arg in the error
%           messages.
% options:  Struct containing contextual parameters.
% 
% OUTPUTS -----------------------------------------------------------------
% arg:      The argument arg, corrected, if possible, to make it satisfy
%           the check.
% -------------------------------------------------------------------------


% Determine which type of error check to perform on arg.
switch(check)
    % Check that arg is a numeric matrix.
    case 'ismatrix'
        matrixcheck(arg, str);
    % Check that arg is a square, numeric matrix.
    case 'issquare'
        matrixcheck(arg, str);
        
        if size(arg, 1) ~= size(arg, 2)
            error(['Argument ', str, ' is not a square matrix!']);
        end
    % Check that arg is a fat, numeric matrix.
    case 'isfat'
        matrixcheck(arg, str);
        
        if size(arg, 1) >= size(arg, 2)
            error(['Argument ', str, ...
                ' is not a fat matrix (more columns than rows)!']);
        end
    % Check that arg is a skinny, numeric matrix.
    case 'isskinny'
        matrixcheck(arg, str);
        
        if size(arg, 1) <= size(arg, 2)
            error(['Argument ', str, ...
                ' is not a skinny matrix (more rows than columns)!']);
        end
    % Check that arg is a numeric vector.
    case 'isvector'
        vectorcheck(arg, str);
    % Check that arg is a numeric row vector. If a column vector, transpose
    % it to get a row vector.
    case 'isrowvector'
        vectorcheck(arg, str);
        
        % Transpose if a column vector.
        if ~isrow(arg)
            arg = arg';
        end
    % Check that arg is a numeric column vector. If a row vector, transpose
    % it to get a column vector.
    case 'iscolumnvector'
        vectorcheck(arg, str);
        
        % Transpose if a row vector.
        if isrow(arg)
            arg = arg';
        end
    % Check that arg is a single number.
    case 'isnumber'
        numbercheck(arg, str);
    % Check that arg is a positive real number.
    case 'ispositivereal'
        numbercheck(arg, str);
        
        % If arg is not real, but is considered positive, make it real.
        if real(arg) > 0
            arg = real(arg);
        else
            error(['Argument ', str, ' is not a positive real number!']);
        end
    % Check that arg is a positive real number.
    case 'isnonnegativereal'
        numbercheck(arg, str);
        
        % If arg is not real, but is considered nonnegative, make it real.
        if real(arg) >= 0
            arg = real(arg);
        else
            error(['Argument ', str, ...
                ' is not a nonnegative real number!']);
        end
    % Check that arg is a single integer.
    case 'isinteger'
        numbercheck(arg, str);
        
        % If arg is a single numeric value, but not a real integer, floor
        % it into one and take the real component.
        if floor(real(arg)) ~= arg
            arg = floor(real(arg));
        end
    % Check that arg is a struct.
    case 'isstruct'
        if ~isstruct(arg)
            error(['Argument ', str, ...
                ' is not a struct! At least pass empty struct!']);
        end
    % Make sure provided slices are correct.
    case 'slices'
        if isfield(options, 'slicelength') && isfield(options, 'workers')
            options.slicelength = errorcheck(options.slicelength, ...
                'isinteger', 'options.slicelength');
            options.workers = errorcheck(options.workers, ...
                'isinteger', 'options.workers');
            arg = slicemaker(arg, options.workers, options.slicelength);
        elseif ~isfield(options, 'slicelength')
            error('Did not provide slicelength in options struct!');
        else
            error('Did not provide workers in options struct!');
        end
end

end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Check that arg is a numeric matrix and return error message with name of
% variable arg if not, describing the issue.

function matrixcheck(arg, str)
% INPUTS ------------------------------------------------------------------
% arg:  The argument we are performing a check on.
% str:  A string of the name of this variable in the context of arguments.
% 
% OUTPUTS -----------------------------------------------------------------
% There are no outputs apart from error messages. If no error message
% appears, the check was successful.
% -------------------------------------------------------------------------

    % Check that arg is a numeric matrix.
    if ~ismatrix(arg) && length(size(arg)) == 2
        error(['Argument ', str, ' is not a matrix!']);
    elseif ~isnumeric(arg)
        error(['Argument ', str, ' is not numeric!']);
    end
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Check that arg is a numeric vector and return error message with name of
% variable arg if not, describing the issue.

function vectorcheck(arg, str)
% Same inputs and outputs as in function matrixcheck.

    % Check that arg is a numeric vector.
    if ~isvector(arg) && length(size(arg)) == 2 && ...
        (size(arg, 1) == 1 || size(arg, 2) == 1)
        error(['Argument ', str, ' is not a vector!']);
    elseif ~isnumeric(arg)
        error(['Argument ', str, ' is not numeric!']);
    end
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Check that arg is a single number and return error message with name of
% variable arg if not, describing the issue.

function numbercheck(arg, str)
% Same inputs and outputs as in function matrixcheck.

    % Check that arg is a single number.
    if ~isnumeric(arg)
        error(['Argument ', str, ' is not numeric!']);
    elseif ~length(size(arg)) == 2 || ~isequal(size(arg), [1, 1])
        error(['Argument ', str, ' is not a single number!']);
    end
end
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Processes and balances a given slice vector over some number of workers
% (if necessary).

function slices = slicemaker(slices, workers, len)
% INPUTS ------------------------------------------------------------------
% slices:   The vector of slices to process and balance.
% workers:  The number of workers in the parallel pool;
% len:      The length of the full vector that is represented in slices.
% 
% OUTPUTS -----------------------------------------------------------------
% slices:   The slice vector to return.
% -------------------------------------------------------------------------


% Check for any weirdness in slices and correct it, if possible. Error
% out, otherwise.
if ~isvector(slices) || ~isnumeric(slices)
    error('Argument slices is not a numeric vector or integer!');
else
    slices = real(floor(slices));
end

% If slices is not an array but a nonzero number, create a vector
% (called slices again) that slices up the problem into segments of
% size equal to this number. Handles nondivisibility by making the last
% segment smaller than slices. This case is essentially for block
% decomposition of the proximal operator.
if length(slices) == 1 && slices > 0
    slicesize = slices;
    slices(1:floor(len/slicesize)) = slicesize;
    slices(ceil(len/slicesize)) = mod(len, slicesize);
% Case that slices is a number equal to zero. This indicates that the
% user wants ADMM to balance out the workload among the workers, and it
% doesn't matter how many elements are in each slice (requires proximal
% operator to be decomposable down to individual components). Thus, we
% turn slices into a balanced vector of slices of a vector of length len.
elseif length(slices) == 1 && slices == 0
    % Case that we don't have divisibility by workers: we must balance.
    if mod(len, workers) ~= 0
        rem = mod(len, workers);
        slicesize = floor(len/workers);
        slices(rem+1:workers) = slicesize;
        slices(1:rem) = slicesize + 1;
    % Case that we don't have to do any balancing of the workload.
    else
        slices(1:workers) = len/workers;
    end
% Slices must be a vector of integers, containing the problem size of
% each slice of x, in order. If the sum of slice sizes does not equal
% the length of x, something must be wrong.
elseif sum(slices) ~= len
    error('The number of parallel slices does not match length of x!');
end

end
% -------------------------------------------------------------------------