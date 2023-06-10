%code for optimization of parameter s 
function [s, Vm, Vs, Tm, Ts] = holdoutCVKernTCSVDD(X, Y, kernel, perc, nrip, intKerPar, C1, C2, C3, C4)
%[s, Vm, Vs, Tm, Ts] = holdoutCVKernTCSVDD(X, Y, kernel, perc, nrip, intKerPar, C1, C2, C3, C4)
% Xtr: the training examples
% Ytr: the training labels
% kernel: the kernel function.
% perc: percentage of the dataset to be used for validation
% nrip: number of repetitions of the test for the parameter
% intKerPar: list of kernel parameters 
%       for example intKerPar = [10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01];
% 
% Output:
% s: kernel parameter that minimize the median of the
%       validation error
% Vm, Vs: median and variance of the validation error for the  parameter
% Tm, Ts: median and variance of the error computed on the training set for the parameter


    nKerPar = numel(intKerPar); 
    
    n = size(X,1);
    ntr = ceil(n*(1-perc));
    
    tmn = zeros(nKerPar, nrip);
    vmn = zeros(nKerPar, nrip);
    
    for rip = 1:nrip
        I = randperm(n);
        Xtr = X(I(1:ntr),:);
        Ytr = Y(I(1:ntr),:);
        Xvl = X(I(ntr+1:end),:);
        Yvl = Y(I(ntr+1:end),:);
        
        
        is = 0;
        for param=intKerPar
            is = is + 1;
            
            [alpha, Rsquared1, Rsquared2, ~, ~, ~, ~, ~, ~]=...
    TC_SVDD_TRAINING(Xtr, Ytr, kernel, param, C1, C2, C3, C4, 'off');

            tmn(is, rip) = ...
    calcErr(TC_SVDD_TEST(Xtr, Ytr, alpha, Xtr, kernel, param, Rsquared1, Rsquared2), Ytr);               
            vmn(is, rip)  = ...
    calcErr(TC_SVDD_TEST(Xtr, Ytr, alpha, Xvl, kernel, param, Rsquared1, Rsquared2), Yvl);
            
    %disp(rip);

        end
        
    end
    
    Tm = median(tmn,2);
    Ts = std(tmn,0,2);
    Vm = median(vmn,2);
    Vs = std(vmn,0,2);
    
    [row, col] = find(Vm <= min(min(Vm)));
    
    s = intKerPar(row(1));
end

