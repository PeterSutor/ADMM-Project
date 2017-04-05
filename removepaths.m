% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Removes paths for the ADMM library. Shouldn't ever be necessary to use
% this feature, as any added paths will be removed once the Matlab session
% ends. Use this only if the functions in this library override any
% existing ones.

function removepaths(quiet)
% INPUTS ------------------------------------------------------------------
% quiet:    Specify whether to suppress output or not. A binary value.
% -------------------------------------------------------------------------


% Handle no input. Default to not being quiet.
if nargin == 0
    quiet = 0;
elseif quiet ~= 1
    quiet = 0;
end

% Figure out which separator to use.
if ispc
    sep = '\';
else
    sep = '/';
end

% Get directory of this function (assumed to be ADMM library's location).
currpath = mfilename('fullpath');       % Get current file path.
currpath = currpath(1:length(currpath) - length(mfilename()));

% Add paths...
rmpath(currpath);                       % Add ADMM library's path.
rmpath([currpath, sep, 'solvers']);     % Add the folder for solvers.
rmpath([currpath, sep, 'testers']);     % Add the folder for testing.
rmpath([currpath, sep, 'examples']);    % Add the folder containing 
                                        % examples of using this library.

% Set global variable that paths have been removed.
global setup;
setup = [];

% Return if no output is specified.
if quiet
    return;
end
                                        
display(['Removed ADMM library and subfolders "solvers", "testers", ', ...
    ' and "examples" from session search path.']);

end
% -------------------------------------------------------------------------