clc; clear all; close all;
%Example of counterfactuals generation based on TC-SVDD with FNR-reduction
%% All features

I=readtable('T2DM_dataset.csv');
%I=I(randperm(size(I, 1)), :);
filepath='';
%% Prova senza comorbidità (solo ipertensione)
%high risk T2DM =1, low risk T2DM = -1
X=[I.Sex, I.Age_1_year_before,double(I.sBP),double(I.BMI),double(I.LDL),double(I.HDL),double(I.TG),double(I.FBS),double(I.Total_Cholesterol), I.HTN];
Y=I.Diabetes;
Y(Y==0)=-1;
nFeat=size(X,2);

%% Stratified Train test split, keeping balancing between T2D and NON T2D also in train and test set
dataset=[X Y];
%rng('default') % for reproducibility
% Cross validation (train: 70%, test: 30%)
cv = cvpartition(Y,'HoldOut',0.3);
idx = cv.test;
% Separate in training and test data
dataTrain = dataset(~idx,:);
dataTest  = dataset(idx,:);
Xtr = dataTrain(:,1:nFeat);
Ytr = dataTrain(:,nFeat+1);
%prop_train=sum(Ytr==-1)%check stratification
Xvl = dataTest(:,1:nFeat);
Yvl = dataTest(:,nFeat+1);
%prop_test=sum(Yvl==-1)
%save training and test set
header = {'Sex','Age','sBP','BMI','LDL','HDL','TG','FBS','Total Cholesterol','HTN','Class'};
body=[Xtr,Ytr];
writecell([header; num2cell(body)],strcat(filepath,'\trainingSet.txt'),'Delimiter','tab');
body=[Xvl,Yvl];
writecell([header; num2cell(body)],strcat(filepath,'\testSet.txt'),'Delimiter','tab');
%% SVDD TRAINING AND OPTIMIZATION
N = size(Xtr,1);
N1 = sum(Ytr == 1); %count class 1
N2 = sum(Ytr == -1); %count class-1
%SVDD parameters
nu1 = 0.05; nu2 = 0.05;
kernel='gaussian';
C1=1;%1/(nu1*N1); 
C2=1;%1/(nu2*N2); 
C3=1/N1; 
C4=1/N2;
%optimization on 1000 points of the training set
X_opt=Xtr(1:1000,:);
Y_opt=Ytr(1:1000,:);
%Optimization of sigma of the gaussian kernel
intKerPar = linspace(0.01,5,30);
[s, Vm, Vs, Tm, Ts] = ... 
  holdoutCVKernTCSVDD(X_opt, Y_opt, kernel, 0.5, 3, intKerPar, C1, C2, C3, C4);%s è il nuovo parametro ottimizzato
% training of the SVDD with the optimized parameters
[alpha, Rsquared1, Rsquared2, a1, a2, x1, x2, SV1, SV2, YSV1, YSV2]=...
    TC_SVDD_TRAINING_NEW(Xtr, Ytr, kernel, s, C1, C2, C3, C4, 'on');

%radius reduction based on FNR (i.e., on S2 only)
treshold=0.08;
[X_star, Y_star, alpha_star, R1_star, R2_star, ...
    a1_star, a2_star, SV1_star, SV2_star, FNR_star] = ...
    FNR_TCSVDD(Xtr, Ytr, alpha, Rsquared1, Rsquared2, kernel, s, C1, C2, C3, C4, treshold)
%% SVDD confusion matrixes
Ytr_pred = ...
    TC_SVDD_TEST(X_star, Y_star, alpha_star, Xtr, kernel, s, R1_star, R2_star);
Yvl_pred = ... 
    TC_SVDD_TEST(X_star, Y_star, alpha_star, Xvl, kernel, s, R1_star, R2_star);

Ptr = nnz(Ytr(:,1)==+1);  
Ntr = nnz(Ytr(:,1)==-1); 
  
Pvl = nnz(Yvl(:,1)==+1);  
Nvl = nnz(Yvl(:,1)==-1); 

M_tr = [Ytr_pred Ytr]; 

M_vl = [Yvl_pred Yvl];
    
disp('Confusion Matrix Training Set:')
TN_tr = sum(M_tr(:,1)==-1 & M_tr(:,2)==-1);
FN_tr = sum(M_tr(:,1)==-1 & M_tr(:,2)==+1);
TP_tr = sum(M_tr(:,1)==+1 & M_tr(:,2)==+1);
FP_tr = sum(M_tr(:,1)==+1 & M_tr(:,2)==-1);
OUT_tr = sum(Ytr_pred == 2); 
CF_tr = [TP_tr FN_tr; FP_tr TN_tr];
disp(CF_tr);
disp('Number of outliers Training Set:')
disp(OUT_tr);
    
disp('Confusion Matrix Validation Set:')
TN_vl = sum(M_vl(:,1)==-1 & M_vl(:,2)==-1);
FN_vl = sum(M_vl(:,1)==-1 & M_vl(:,2)==+1);
TP_vl = sum(M_vl(:,1)==+1 & M_vl(:,2)==+1);
FP_vl = sum(M_vl(:,1)==+1 & M_vl(:,2)==-1);
OUT_vl = sum(Yvl_pred == 2); 
CF_vl = [TP_vl FN_vl; FP_vl TN_vl];
disp(CF_vl);
disp('Number of outliers Training Set:')
disp(OUT_vl);

header1 = {'Feats','TP','FN','FP','TN','OUTLIERS'};
features={'All'};
new_row_tr=[features num2cell([TP_tr, FN_tr, FP_tr, TN_tr, OUT_tr])];
new_row_vl=[features num2cell([TP_vl, FN_vl, FP_vl, TN_vl, OUT_vl])];
%save training performance
if(~isfile(strcat(filepath,'SVDD_training_perf.txt')))
    SVDD_perf_tr = [header1; new_row_tr];  
else
    SVDD_perf_tr = new_row_tr;  
end
writecell(SVDD_perf_tr ,strcat('SVDD_training_perf.txt'),'Delimiter','tab','WriteMode','append');
%save test performance
if(~isfile(strcat(filepath,'SVDD_test_perf.txt')))
    SVDD_perf_ts = [header1; new_row_vl];
else
    SVDD_perf_ts = new_row_vl;
end
writecell(SVDD_perf_ts ,strcat('SVDD_test_perf.txt'),'Delimiter','tab','WriteMode','append');
%% sampling of SVDD boundary and internal points using Halton quasi-random sampling
Csample = [];
Xn1 = [];

X=Xtr;
p_halton=haltonset(1);
halton_seq=net(p_halton,50000); %halton sequence (50000 values between 0 and 1)

%sampling region S1 (here not used for counterfactuals, but used only for
%extracting decision rules)
while (length(Xn1)<10000)
    
    X1 = min(X(:,1))+(max(X(:,1))-min(X(:,1)))*randsample(halton_seq,1000);
    X2 = min(X(:,2))+(max(X(:,2))-min(X(:,2)))*randsample(halton_seq,1000);
    X3 = min(X(:,3))+(max(X(:,3))-min(X(:,3)))*randsample(halton_seq,1000);
    X4 = min(X(:,4))+(max(X(:,4))-min(X(:,4)))*randsample(halton_seq,1000);
    X5 = min(X(:,5))+(max(X(:,5))-min(X(:,5)))*randsample(halton_seq,1000);
    X6 = min(X(:,6))+(max(X(:,6))-min(X(:,6)))*randsample(halton_seq,1000);
    X7 = min(X(:,7))+(max(X(:,7))-min(X(:,7)))*randsample(halton_seq,1000);
    X8 = min(X(:,8))+(max(X(:,8))-min(X(:,8)))*randsample(halton_seq,1000);
    X9 = min(X(:,9))+(max(X(:,9))-min(X(:,9)))*randsample(halton_seq,1000);
    X10 = min(X(:,10))+(max(X(:,10))-min(X(:,10)))*randsample(halton_seq,1000);
  
    C_camp = [X1 X2 X3 X4 X5 X6 X7 X8 X9 X10];
    
    Tts1 = TestObject_N(X_star, Y_star, x1, C_camp, kernel, s); %dist from a1
    Tts2 = TestObject_N(X_star, -Y_star, x2, C_camp, kernel, s); %dist from a2
        
    Xn = C_camp(((Tts1-R1_star<0).*(Tts2-R2_star>0)==1),:); %points in class 1
    
    Xn1 = [Xn1; Xn];
    disp(length(Xn1))
end

yXn1 = ...
            TC_SVDD_TEST(X_star, Y_star, alpha_star, Xn1, kernel, s, R1_star, R2_star);

Xn1 = [Xn1, yXn1];


%%
%sampling region S2
Xn2 = [];
X=Xtr;
while (length(Xn2)<10000)
    
    X1 = min(X(:,1))+(max(X(:,1))-min(X(:,1)))*randsample(halton_seq,1000);
    X2 = min(X(:,2))+(max(X(:,2))-min(X(:,2)))*randsample(halton_seq,1000);
    X3 = min(X(:,3))+(max(X(:,3))-min(X(:,3)))*randsample(halton_seq,1000);
    X4 = min(X(:,4))+(max(X(:,4))-min(X(:,4)))*randsample(halton_seq,1000);
    X5 = min(X(:,5))+(max(X(:,5))-min(X(:,5)))*randsample(halton_seq,1000);
    X6 = min(X(:,6))+(max(X(:,6))-min(X(:,6)))*randsample(halton_seq,1000);
    X7 = min(X(:,7))+(max(X(:,7))-min(X(:,7)))*randsample(halton_seq,1000);
    X8 = min(X(:,8))+(max(X(:,8))-min(X(:,8)))*randsample(halton_seq,1000);
    X9 = min(X(:,9))+(max(X(:,9))-min(X(:,9)))*randsample(halton_seq,1000);
    X10 = min(X(:,10))+(max(X(:,10))-min(X(:,10)))*randsample(halton_seq,1000);
    
    C_camp = [X1 X2 X3 X4 X5 X6 X7 X8 X9 X10];
    
    Tts1 = TestObject_N(X_star, Y_star, x1, C_camp, kernel, s); %dist from a1
    Tts2 = TestObject_N(X_star, -Y_star, x2, C_camp, kernel, s); %dist from a2
      
    Xn = C_camp(((Tts2-R2_star<0).*(Tts1-R1_star>0)==1),:); %points in class 2
  
    Xn2 = [Xn2;Xn];
    %Xn2_1 = unique([Xn2_1; Xn],'rows');
    disp(length(Xn2))
end

yXn2 = ...
            TC_SVDD_TEST(X_star, Y_star, alpha_star, Xn2, kernel, s, R1_star, R2_star);

Xn2 = [Xn2, yXn2];

Csample = unique([Csample; [Xn1;Xn2]],'rows');

Xn1 = Csample(Csample(:,nFeat+1)==1,:); %punti della regione 1
Xn2 = Csample(Csample(:,nFeat+1)==-1,:); %punti della regione -1
%save sampled svdd points
output = [header; num2cell(Csample)];
writecell(output,'SVDD_points.txt','Delimiter','tab');

%% generating countefactuals
X_real = [Xvl, Yvl];
X_pred = [Xvl, Yvl_pred];
X_pred = X_pred(X_real(:,nFeat+1) == 1,:);% people at high risk of T2DM
X_T2D = X_pred(X_pred(:,nFeat+1) ~= -1,:); %people at high risk of T2DM that are not predicted as low risk (i.e., predicted as 1 or 2)->more data wrt to 1

XC_T2D = []; %counterfactuals, low T2DM
Xn2=Xn2(:,1:nFeat);
X_candidates=[];
dist=[];
next_close_points=[];
conta_counter=0;
for i = 1 : size(X_T2D,1) %for each factual
    
    next_close_points=[];
    for j=1: size(Xn2,1) %check all constrained candidate counterfactuals
        if (abs(Xn2(j,1)-X_T2D(i,1))<=0.5 && abs(Xn2(j,2)-X_T2D(i,2))<=1 && (X_T2D(i,10)-Xn2(j,10))<=0.5) % age, gender and presence of hypertension are kept fixed
            X_candidates=[X_candidates;Xn2(j,1:nFeat)]; 
        end
    end
     if isempty(X_candidates)==1 %if there are no candidate counterfactuals (respecting constraints)
        A=[0,0,0,0,0,0,0,0,0,0];
        dist_i = -99;

        dist = [dist;dist_i];
        XC_T2D = [XC_T2D; A];
     else
        
        z = X_T2D(i,1:nFeat);
        K = KernelMatrix(X_candidates,z,kernel, s);
        sq_dist = 2*(1-K); % vector of distances between factuals and candidate counterfactuals
        
        sq_minimum = min(sq_dist);
        [indexOpt,j] = find(sq_dist == sq_minimum); % index corresponding to minimum distance

        XC_T2D = [XC_T2D;X_candidates(indexOpt(1),:)];
        zC = X_candidates(indexOpt(1),:);

        dist_i = TestObject_N(X_star, Y_star, x1, zC, kernel, s)-R1_star;

        dist = [dist;dist_i];
      end
     X_candidates=[];

end
yC_CKD =TC_SVDD_TEST(X_star, Y_star, alpha_star, XC_T2D, kernel, s, R1_star, R2_star); %counterfactual predicted class
XC_T2D = [XC_T2D, yC_CKD];

%% save factuals and relative counterfactuals
%save factuals
body=[X_T2D(:,1),X_T2D(:,2),X_T2D(:,3),X_T2D(:,4),X_T2D(:,5),X_T2D(:,6),X_T2D(:,7),X_T2D(:,8),X_T2D(:,9),X_T2D(:,10),X_T2D(:,11)];
writecell([header; num2cell(body)],strcat(filepath,'\factuals.txt'),'Delimiter','tab');

%save corresponding counterfactuals
body=[XC_T2D(:,1),XC_T2D(:,2),XC_T2D(:,3),XC_T2D(:,4),XC_T2D(:,5),XC_T2D(:,6),XC_T2D(:,7),XC_T2D(:,8),XC_T2D(:,9),XC_T2D(:,10),XC_T2D(:,11)];
writecell([header; num2cell(body)],strcat(filepath,'\counterfactuals.txt'),'Delimiter','tab'); 
