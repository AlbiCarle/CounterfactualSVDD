%code for FNR reduction
function [X_star, Y_star, alpha_star, Rsquared1_star, Rsquared2_star, ...
    a1_star, a2_star, SV1_star, SV2_star, FNR_star] = ...
    FNR_TCSVDD(X, Y, alpha, Rsquared1, Rsquared2, kernel, param, C1, C2, C3, C4, threshold)
maxiter=1000;

i=0;

m = size(X,2);
y_i= TC_SVDD_TEST(X, Y, alpha, X, kernel, param, Rsquared1, Rsquared2);


while(i<maxiter)
   
    i=i+1;
    
    X_pred_i=[X,Y,y_i];
    
    XP_i = X_pred_i(X_pred_i(:,m+2)==1,(1:m)); %predicted pos
    disp(size(XP_i))
    XN_i = X_pred_i(X_pred_i(:,m+2)==-1,(1:m)); %predicted neg
    disp(size(XN_i))
    X_i = [XP_i;XN_i];
    YP_i=X_pred_i(X_pred_i(:,m+2)==1,m+1); %real label associated to predicted pos
    YN_i=X_pred_i(X_pred_i(:,m+2)==-1,m+1); %real label associated to predicted neg
    Y_i = [YP_i;YN_i];
   [alpha, Rsquared1_i, Rsquared2_i, a1_i, a2_i, ~, ~, SV1_i, SV2_i, ~, ~]=...
    TC_SVDD_TRAINING_NEW(X_i, Y_i, kernel, param, C1, C2, C3, C4, 'off');
    
    y_i = ...
        TC_SVDD_TEST(X_i, Y_i, alpha, X, kernel, param, Rsquared1_i, Rsquared2_i);
    
    M_i = [y_i Y]; %predicted and real labels
    
    P = nnz(Y_i(:,1)==+1);  
    N = nnz(Y_i(:,1)==-1); 
            
    TP = sum(M_i(:,1)==+1 & M_i(:,2)==+1); %true positives: real positive points that are correctly predicted
    FN = sum(M_i(:,1)==-1 & M_i(:,2)==1); %false negatives: real positive points predicted as negative
   
    
    FNR_i = (FN)/(FN+TP);
    disp('FNR')
    disp(FNR_i)
    if(FNR_i<threshold) %if the FNR is below the threshold
        disp("found")
        disp(FNR_i)
        disp(i)
        break;
    end
    
    disp(i);
    
end

X_star = X_i; 
Y_star = Y_i;
alpha_star = alpha;

Rsquared1_star = Rsquared1_i;
Rsquared2_star = Rsquared2_i;

a1_star = a1_i;
a2_star = a2_i;

SV1_star = SV1_i;
SV2_star = SV2_i;

FNR_star = FNR_i;







