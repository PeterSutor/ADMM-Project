% -------------------------------------------------------------------------
% Trains data on SVM solver with unwrapped ADMM and then performs one round
% of cross validation on test data from MNIST. Uses both Hinge and 0-1 loss
% functions.
function svm_test_mnist(C, rho, testsubsets, trainsubsets)
% INPUTS (if data is already loaded, specified by loaddata):
% C:                    Regularization parameter.
% rho:                  Step size in ADMM.
% testsubsets:          Size of the testing subset to take (out of 10000).
% trainsubsets:         Size of the training subsets to take (out of 60000).
% -------------------------------------------------------------------------

% Load raw images and labels from files. There are at most 10000 testing
% images and 60000 training images.
[testIm, testLab] = ...
    readMNIST('MNIST/t10k-images.idx3-ubyte', 'MNIST/t10k-labels.idx1-ubyte', 10000, 0);
[trainIm, trainLab] = readMNIST('MNIST/train-images.idx3-ubyte', ...
    'MNIST/train-labels.idx1-ubyte', 60000, 0);

% Length of them.
m = length(testLab);
n = length(trainLab);

% Initialize.
testVecAll = zeros(m, 400);
trainVecAll = zeros(n, 400);

% Reshape into vectorized images.
for i = 1:m
    testVecAll(i, :) = reshape(testIm(:, :, i)', 1, 400);
end

% Reshape into vectorized images.
for i = 1:n
    trainVecAll(i, :) = reshape(trainIm(:, :, i)', 1, 400);
end

testTemp = datasample([testVecAll testLab], testsubsets);
trainTemp = datasample([trainVecAll trainLab], trainsubsets);

testVec = testTemp(:, 1:end-1);
testLab = testTemp(:, end);
trainVec = trainTemp(:, 1:end-1);
trainLab = trainTemp(:, end);

m = length(testLab);
n = length(trainLab);

% Stores error percentage results.
result = zeros(10, 4);

% Loop over every digit and train for it.
for i = 1:10
    % Train for current digit i-1, for both Hinge and 0-1 loss functions.
    [xhin, x01] = trainForDigit(C, rho, trainVec, trainLab, i - 1);
    
    % Form positive/negative labels for classifying.
    trainLabAlt = (trainLab == i - 1)*2 - 1;
    testLabAlt = (testLab == i - 1)*2 - 1;
    
    % Store percentages of incorrect results (errors).
    result(i, 1) = sum((1 - trainLabAlt.*(trainVec*xhin)) > 0)/n*100.0;
    result(i, 2) = sum((1 - trainLabAlt.*(trainVec*x01)) > 0)/n*100.0;
    result(i, 3) = sum((1 - testLabAlt.*(testVec*xhin)) > 0)/m*100.0;
    result(i, 4) = sum((1 - testLabAlt.*(testVec*x01)) > 0)/m*100.0;
end

% Headers for output.
fprintf('\nError Percentages:\n\n');
fprintf('%s\t%s\t%s\t%s\t%s\n%', 'Digit', ...
    'Hinge (Train)', '0-1 (Train)', 'Hinge (Test)', '0-1 (Test)');

% Print results.
for i = 1:10
    fprintf('%d\t\t%2.4f\t\t\t%2.4f\t\t%2.4f\t\t\t%2.4f\n', i - 1, ...
        result(i, 1), result(i, 2), result(i, 3), result(i, 4));
end

end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Train for given digit using labels ell and data D.
function [xhin, x01] = trainForDigit(C, rho, D, ell, digit)
% INPUTS:
% D         Data matrix. Single row is a row-vectorized image.
% ell       Labels for actual digits shown in image.
% digit     The digit to train for.
% -------------------------------------------------------------------------
% OUPUTS:
% xhin      The trained solution for the Hinge loss function.
% x01       The trained solution for the 0-1 loss function.
% -------------------------------------------------------------------------

% Number of images as data.
a = length(ell);

% Form positive and negative weight for labels based on digit to train for.
for i = 1:a
    if (ell(i) == digit)
        ell(i) = 1;
    else
        ell(i) = -1;
    end
end

options.rho = rho;
options.maxiters = 250;
options.domaxiters = 1;

% Use unwrapped ADMM to obtain solutions.
resultshinge = linearSVM(D, ell, C, 1, options);
results01 = linearSVM(D, ell, C, 0, options);

xhin = resultshinge.xopt;
x01 = results01.xopt;

% N1 = resultshinge.steps;
% N2 = results01.steps;
%
% % Figure for the objective value at each iteration.
% figure;
% subplot(3, 1, 1);
% plot(1:N1, resultshinge.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
% title('Plot of hinge-loss value for each iteration');
% ylabel('Objective'); 
% xlabel('Iteration k');

% Figure of primal residual norm.
% figure;
% subplot(3, 1, 2);
% semilogy(1:N1, max(1e-8, resultshinge.pnorm), 'k', ...
%     1:N1, resultshinge.perr, 'k--',  'LineWidth', 2);
% title('Plot of Primal Residual Norm for Hinge Loss');
% ylabel('||Ax + Bz - c||_2');
% xlabel('Iteration k');
% 
% % Figure of dual residual norm
% subplot(3, 1, 3);
% semilogy(1:N1, max(1e-8, resultshinge.dnorm), 'k', ...
%     1:N1, resultshinge.derr, 'k--', 'LineWidth', 2);
% title('Plot of Dual Residual Norm for Hinge Loss');
% ylabel('||rho*A^TB(z^k-z^{k-1})||_2'); 
% xlabel('Iteration k');
% 
% 
% % Figure of primal residual norm.
% figure
% subplot(3, 1, 2);
% semilogy(1:N2, max(1e-8, results01.pnorm), 'k', ...
%     1:N2, results01.perr, 'k--',  'LineWidth', 2);
% title('Plot of Primal Residual Norm for 0-1 Loss');
% ylabel('||Ax + Bz - c||_2');
% xlabel('Iteration k');
% 
% % Figure of dual residual norm
% subplot(3, 1, 3);
% semilogy(1:N2, max(1e-8, results01.dnorm), 'k', ...
%     1:N2, results01.derr, 'k--', 'LineWidth', 2);
% title('Plot of Dual Residual Norm for 0-1 Loss');
% ylabel('||rho*A^TB(z^k-z^{k-1})||_2'); 
% xlabel('Iteration k');

end
% -------------------------------------------------------------------------



% readMNIST by Siddharth Hegde
%
% Description:
% Read digits and labels from raw MNIST data files
% File format as specified on http://yann.lecun.com/exdb/mnist/
% Note: The 4 pixel padding around the digits will be remove
%       Pixel values will be normalised to the [0...1] range
%
% Usage:
% [imgs labels] = readMNIST(imgFile, labelFile, readDigits, offset)
%
% Parameters:
% imgFile = name of the image file
% labelFile = name of the label file
% readDigits = number of digits to be read
% offset = skips the first offset number of digits before reading starts
%
% Returns:
% imgs = 20 x 20 x readDigits sized matrix of digits
% labels = readDigits x 1 matrix containing labels for each digit
%
function [imgs, labels] = readMNIST(imgFile, labelFile, readDigits, offset)
    
    % Read digits
    fid = fopen(imgFile, 'r', 'b');
    header = fread(fid, 1, 'int32');
    if header ~= 2051
        error('Invalid image file header');
    end
    count = fread(fid, 1, 'int32');
    if count < readDigits+offset
        error('Trying to read too many digits');
    end
    
    h = fread(fid, 1, 'int32');
    w = fread(fid, 1, 'int32');
    
    if offset > 0
        fseek(fid, w*h*offset, 'cof');
    end
    
    imgs = zeros([h w readDigits]);
    
    for i=1:readDigits
        for y=1:h
            imgs(y,:,i) = fread(fid, w, 'uint8');
        end
    end
    
    fclose(fid);

    % Read digit labels
    fid = fopen(labelFile, 'r', 'b');
    header = fread(fid, 1, 'int32');
    if header ~= 2049
        error('Invalid label file header');
    end
    count = fread(fid, 1, 'int32');
    if count < readDigits+offset
        error('Trying to read too many digits');
    end
    
    if offset > 0
        fseek(fid, offset, 'cof');
    end
    
    labels = fread(fid, readDigits, 'uint8');
    fclose(fid);
    
    % Calc avg digit and count
    imgs = trimDigits(imgs, 4);
    imgs = normalizePixValue(imgs);
    %[avg num stddev] = getDigitStats(imgs, labels);
    
end

function digits = trimDigits(digitsIn, border)
    dSize = size(digitsIn);
    digits = zeros([dSize(1)-(border*2) dSize(2)-(border*2) dSize(3)]);
    for i=1:dSize(3)
        digits(:,:,i) = digitsIn(border+1:dSize(1)-border, border+1:dSize(2)-border, i);
    end
end

function digits = normalizePixValue(digits)
    digits = double(digits);
    for i=1:size(digits, 3)
        digits(:,:,i) = digits(:,:,i)./255.0;
    end
end