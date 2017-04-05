% Coded by:     Peter Sutor Jr.
% Last edit:    4/10/2016
% -------------------------------------------------------------------------
% DESCRIPTION -------------------------------------------------------------
%
% Sets up paths for the ADMM library. Useful for if you plan on calling 
% functions from this library externally, from other folders. If you will 
% only use the ADMM functions in the main folder (this one), this action is
% not necessary. If needed, other functions will try to set up their paths
% on their own. Note that these paths are setup only for this session.

function setuppaths(quiet)
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
addpath(currpath);                      % Add ADMM library's path.
addpath([currpath, sep, 'solvers']);    % Add the folder for solvers.
addpath([currpath, sep, 'testers']);    % Add the folder for testing.
addpath([currpath, sep, 'examples']);   % Add the folder containing 
                                        % examples of using this library.

% Set global variable that paths have been setup.
global setup;
setup = 1;

% Return if no output is specified.
if quiet
    return;
end

display(['Added ADMM library and subfolders "solvers", "testers", and', ...
    ' "examples" to session search path.']);

end
% -------------------------------------------------------------------------