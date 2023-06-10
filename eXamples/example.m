clc; clear all; close all;

%% All features

I = readmatrix('LogMetrics.txt');

I = I(randperm(size(I, 1)), :);
I1 = I(I(:,8)==1,:);
I2 = I(I(:,8)==0,:);

I21 = I(randperm(size(I1, 1)), :);

I = [I1;I21];

X = I(:,1:7);

MAX1 = 10;
MAX2 = abs(max(I(:,2)));
MAX3 = abs(max(I(:,3)));
MAX4 = abs(max(I(:,4)));
MAX5 = abs(max(I(:,5)));
MAX6 = abs(max(I(:,6)));
MAX7 = abs(max(I(:,7)));

MAX = [10, MAX2, MAX3, MAX4, MAX5, MAX6, MAX7];

X1 = X(:,1)/10;
X2 = X(:,2)/abs(max(X(:,2)));
X3 = X(:,3)/abs(max(X(:,3)));
X4 = X(:,4)/abs(max(X(:,4)));
X5 = X(:,5)/abs(max(X(:,5)));
X6 = X(:,6)/abs(max(X(:,6)));
X7 = X(:,7)/abs(max(X(:,7)));

X = [X1, X2, X3, X4, X5, X6, X7];

%[X, mu, sigma] = zscore(X); for normalisation with Zscore

Y = I(:,8);
Y(Y==0) = -1;

m = size(X,2); n = size(X,1);

ntr = ceil(n*(1-0.4));

c = randperm(n);

Xlearn = X(c(1:ntr),:);
Ylearn = Y(c(1:ntr),:);

    n2 = size(Xlearn,1);
    ntr2 = ceil(n2*(1-0.4));
    c2 = randperm(n2);

    Xtr = Xlearn(c2(1:ntr2),:);
    Ytr = Ylearn(c2(1:ntr2),:);
    Xvl = Xlearn(c2(ntr2+1:end),:);
    Yvl = Ylearn(c2(ntr2+1:end),:);

Xts = X(c(ntr+1:end),:);
Yts = Y(c(ntr+1:end),:);

header1 = {'N','F0','m','d_ms','d0','v0','prob','class'};
PlatTest=[header1; num2cell([Xts,Yts])];

writecell(PlatTest,'./example_Test_.txt','Delimiter','tab');

feat={'$N$','$F$','$m$','$d_{ms}$','$d_0$','$v_0$','$p$'};

%% For showing the scatterplots of the features

k1=0;
figure(1)

for i = 1:m
    for j = i+1:m
        k1=k1+1;
    subplot(3,7,k1)
    gscatter(Xtr(:,i), Xtr(:,j), Ytr,'br','.',[8 8]);
    xlabel(feat(i), 'Interpreter', 'latex')
    ylabel(feat(j), 'Interpreter', 'latex')
    a = feat(i); b = feat(j);
    title([feat{i} ' vs ' feat{j}],'Interpreter','latex','FontSize', 14)
    legend off
    end
end
legend('non-collision','collision')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

intKerPar = linspace(1,5,100);

opt = 'yes';

if isequal(opt,'yes')

[s, Vm, Vs, Tm, Ts] = ... 
 holdoutCVKernTCSVDD(X_opt, Y_opt, kernel, 0.5, 6, intKerPar, C1, C2, C3, C4);%s è il nuovo parametro ottimizzato

else
    s = 1;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training of the SVDD with the optimized parameters
[alpha, Rsquared1, Rsquared2, a1, a2, x1, x2, SV1, SV2, YSV1, YSV2]=...
    TC_SVDD_TRAINING_NEW(Xtr, Ytr, kernel, s, C1, C2, C3, C4, 'on');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SVDD confusion matrixes

Ytr_pred = ...
    TC_SVDD_TEST(Xtr, Ytr, alpha, Xtr, kernel, s, Rsquared1, Rsquared2);
Yvl_pred = ... 
    TC_SVDD_TEST(Xtr, Ytr, alpha, Xvl, kernel, s, Rsquared1, Rsquared2);

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

ACC_tr = (TP_tr+TN_tr)/(TP_tr+TN_tr+FP_tr+FN_tr);
disp('Accuracy Training Set:')
disp(ACC_tr);
    
disp('Confusion Matrix Validation Set:')

TN_vl = sum(M_vl(:,1)==-1 & M_vl(:,2)==-1);
FN_vl = sum(M_vl(:,1)==-1 & M_vl(:,2)==+1);
TP_vl = sum(M_vl(:,1)==+1 & M_vl(:,2)==+1);
FP_vl = sum(M_vl(:,1)==+1 & M_vl(:,2)==-1);
OUT_vl = sum(Yvl_pred == 2); 

CF_vl = [TP_vl FN_vl; FP_vl TN_vl];
disp(CF_vl);

disp('Number of outliers Validation Set:')
disp(OUT_vl);

ACC_vl = (TP_vl+TN_vl)/(TP_vl+TN_vl+FP_vl+FN_vl);
disp('Accuracy Validation Set:')
disp(ACC_tr);

disp('Sensitivity:')
Se_vl = (TP_vl)/(TP_vl+FN_vl);
disp(Se_vl);

disp('Specificity:')
Sp_vl = (TN_vl)/(TN_vl+FP_vl);
disp(Sp_vl);

filepath='.\';
header1 = {'Feats','TP','FN','FP','TN','OUTLIERS'};
features={'All'};
new_row_tr=[features num2cell([TP_tr, FN_tr, FP_tr, TN_tr, OUT_tr])];
new_row_vl=[features num2cell([TP_vl, FN_vl, FP_vl, TN_vl, OUT_vl])];
if(~isfile(strcat(filepath,'train_prediction_metrics.txt')))
    SVDD_perf_tr = [header1; new_row_tr];
   
else
    SVDD_perf_tr = new_row_tr;
   
end

if(~isfile(strcat(filepath,'val_prediction_metrics.txt')))
    SVDD_perf_ts = [header1; new_row_vl];
else
    SVDD_perf_ts = new_row_vl;
end

writecell(SVDD_perf_tr ,strcat('SVDD_training_perf.txt'),'Delimiter','tab','WriteMode','append');
writecell(SVDD_perf_ts ,strcat('SVDD_val_perf.txt'),'Delimiter','tab','WriteMode','append');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Sampling of SVDD boundary points
% for sampling the points around the boundary decision

Xsample = [];
X = Xtr;
while (length(Xsample)<2500)
    
    X1 = randi([min(X(:,1))*MAX1 max(X(:,1)*MAX1)],1,1000);
    X2 = min(X(:,2))+(max(X(:,2))-min(X(:,2)))*rand(1000,1);
    X3 = min(X(:,3))+(max(X(:,3))-min(X(:,3)))*rand(1000,1);
    X4 = min(X(:,4))+(max(X(:,4))-min(X(:,4)))*rand(1000,1);
    X5 = min(X(:,5))+(max(X(:,5))-min(X(:,5)))*rand(1000,1);
    X6 = min(X(:,6))+(max(X(:,6))-min(X(:,6)))*rand(1000,1);
    X7 = min(X(:,7))+(max(X(:,7))-min(X(:,7)))*rand(1000,1);

    X_camp = [X1'/MAX1 X2 X3 X4 X5 X6 X7];

    Tts1 = TestObject_N(Xtr, Ytr, x1, X_camp, kernel, s);
    Tts2 = TestObject_N(Xtr, -Ytr, x2, X_camp, kernel, s);
    
    Xn1 = X_camp(abs(Rsquared1-Tts1)<0.0001,:);
    Xn2 = X_camp(abs(Rsquared2-Tts2)<0.0001 ,:);

    Xn = [Xn1; Xn2];
    Xsample = [Xsample; Xn];

    disp(length(Xsample))
end

%predicted class related to sampled points

ysample = ...
            TC_SVDD_TEST(Xtr, Ytr, alpha, Xsample, kernel, s, Rsquared1, Rsquared2);
Xnew=[Xsample,ysample]; %associate to each sample its predicted class
header = {'N','F0','m','d_ms','d0','v0','prob','class'};
output = [header; num2cell([Xnew(:,1:7).*MAX, Xnew(:,8)])];
writecell(output,'CAAC_EDGE_RULEX.txt','Delimiter','tab');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COUNTERFACTUALS GENERATION: building the sampling dataset

Csample = [];
Xn1_1 = [];

X=Xtr;

while (length(Csample)<100000) %300000

while (length(Xn1_1)<10000)
    
    X1 = randi([min(X(:,1))*MAX1 max(X(:,1)*MAX1)],1,1000);
    X2 = min(X(:,2))+(max(X(:,2))-min(X(:,2)))*rand(1000,1);
    X3 = min(X(:,3))+(max(X(:,3))-min(X(:,3)))*rand(1000,1);
    X4 = min(X(:,4))+(max(X(:,4))-min(X(:,4)))*rand(1000,1);
    X5 = min(X(:,5))+(max(X(:,5))-min(X(:,5)))*rand(1000,1);
    X6 = min(X(:,6))+(max(X(:,6))-min(X(:,6)))*rand(1000,1);
    X7 = min(X(:,7))+(max(X(:,7))-min(X(:,7)))*rand(1000,1);

    C_camp = [X1'/MAX1 X2 X3 X4 X5 X6 X7];
    
    Tts1 = TestObject_N(Xtr, Ytr, x1, C_camp, kernel, s); %dist from a1
    Tts2 = TestObject_N(Xtr, -Ytr, x2, C_camp, kernel, s); %dist from a2
        
    Xn1 = C_camp(((Tts1-Rsquared1<0).*(Tts2-Rsquared2>0)==1),:); %points in class 1
    %Xn2 = C_camp(((Tts2-Rsquared2<0).*(Tts1-Rsquared1>0)==1),:); %points in class 2 
    
    %Xn = [Xn1; Xn2];
    Xn = Xn1;
    Xn1_1 = unique([Xn1_1; Xn],'rows');
    disp(length(Xn1_1))
end

yXn1 = TC_SVDD_TEST(Xtr, Ytr, alpha, Xn1_1, kernel, s, Rsquared1, Rsquared2);

Xn1_11 = [Xn1_1, yXn1];

Xn1_2 = [];

X=Xtr;

while (length(Xn1_2)<10000)
    
    X1 = randi([min(X(:,1))*MAX1 max(X(:,1)*MAX1)],1,1000);
    X2 = min(X(:,2))+(max(X(:,2))-min(X(:,2)))*rand(1000,1);
    X3 = min(X(:,3))+(max(X(:,3))-min(X(:,3)))*rand(1000,1);
    X4 = min(X(:,4))+(max(X(:,4))-min(X(:,4)))*rand(1000,1);
    X5 = min(X(:,5))+(max(X(:,5))-min(X(:,5)))*rand(1000,1);
    X6 = min(X(:,6))+(max(X(:,6))-min(X(:,6)))*rand(1000,1);
    X7 = min(X(:,7))+(max(X(:,7))-min(X(:,7)))*rand(1000,1);

    C_camp = [X1'/MAX1 X2 X3 X4 X5 X6 X7];
    
    Tts1 = TestObject_N(Xtr, Ytr, x1, C_camp, kernel, s); %dist from a1
    Tts2 = TestObject_N(Xtr, -Ytr, x2, C_camp, kernel, s); %dist from a2
        
    Xn1 = C_camp(((Tts1-Rsquared1<0).*(Tts2-Rsquared2>0)==1),:); %points in class 1
    %Xn2 = C_camp(((Tts2-Rsquared2<0).*(Tts1-Rsquared1>0)==1),:); %points in class 2
    
    %Xn = [Xn1; Xn2];
    Xn = Xn1;
    Xn1_2 = unique([Xn1_2; Xn],'rows');
    disp(length(Xn1_2))
end

yXn1 = TC_SVDD_TEST(Xtr, Ytr, alpha, Xn1_2, kernel, s, Rsquared1, Rsquared2);

Xn1_12 = [Xn1_2, yXn1];

Xn1 = [Xn1_11;Xn1_12];

%%%

Xn2_1 = [];
X = Xtr;
while (length(Xn2_1)<10000)
    
    X1 = randi([min(X(:,1))*MAX1 max(X(:,1)*MAX1)],1,1000);
    X2 = min(X(:,2))+(max(X(:,2))-min(X(:,2)))*rand(1000,1);
    X3 = min(X(:,3))+(max(X(:,3))-min(X(:,3)))*rand(1000,1);
    X4 = min(X(:,4))+(max(X(:,4))-min(X(:,4)))*rand(1000,1);
    X5 = min(X(:,5))+(max(X(:,5))-min(X(:,5)))*rand(1000,1);
    X6 = min(X(:,6))+(max(X(:,6))-min(X(:,6)))*rand(1000,1);
    X7 = min(X(:,7))+(max(X(:,7))-min(X(:,7)))*rand(1000,1);

    C_camp = [X1'/MAX1 X2 X3 X4 X5 X6 X7];
    
    Tts1 = TestObject_N(Xtr, Ytr, x1, C_camp, kernel, s); %dist from a1
    Tts2 = TestObject_N(Xtr, -Ytr, x2, C_camp, kernel, s); %dist from a2
        
    %Xn1 = C_camp(((Tts1-Rsquared1<0).*(Tts2-Rsquared2>0)==1),:); %points in class 1
    Xn2 = C_camp(((Tts2-Rsquared2<0).*(Tts1-Rsquared1>0)==1),:); %points in class 2
    
    %Xn = [Xn1; Xn2];
    Xn = Xn2;
    Xn2_1 = unique([Xn2_1; Xn],'rows');
    disp(length(Xn2_1))
end

yXn2 = TC_SVDD_TEST(Xtr, Ytr, alpha, Xn2_1, kernel, s, Rsquared1, Rsquared2);

Xn2_11 = [Xn2_1, yXn2];

Xn2_2 = [];
X = Xtr;

while (length(Xn2_2)<10000)
    
    X1 =  randi([min(X(:,1))*MAX1 max(X(:,1)*MAX1)],1,1000);
    X2 = min(X(:,2))+(max(X(:,2))-min(X(:,2)))*rand(1000,1);
    X3 = min(X(:,3))+(max(X(:,3))-min(X(:,3)))*rand(1000,1);
    X4 = min(X(:,4))+(max(X(:,4))-min(X(:,4)))*rand(1000,1);
    X5 = min(X(:,5))+(max(X(:,5))-min(X(:,5)))*rand(1000,1);
    X6 = min(X(:,6))+(max(X(:,6))-min(X(:,6)))*rand(1000,1);
    X7 = min(X(:,7))+(max(X(:,7))-min(X(:,7)))*rand(1000,1);

    C_camp = [X1'/MAX1 X2 X3 X4 X5 X6 X7];
    
    Tts1 = TestObject_N(Xtr, Ytr, x1, C_camp, kernel, s); %dist from a1
    Tts2 = TestObject_N(Xtr, -Ytr, x2, C_camp, kernel, s); %dist from a2
        
    %Xn1 = C_camp(((Tts1-Rsquared1<0).*(Tts2-Rsquared2>0)==1),:); %points in class 1
    Xn2 = C_camp(((Tts2-Rsquared2<0).*(Tts1-Rsquared1>0)==1),:); %points in class 2
    
    %Xn = [Xn1; Xn2];
    Xn = Xn2;
    Xn2_2 = unique([Xn2_2; Xn],'rows');
    disp(length(Xn2_2))
end

yXn2 = TC_SVDD_TEST(Xtr, Ytr, alpha, Xn2_2, kernel, s, Rsquared1, Rsquared2);

Xn2_12 = [Xn2_2, yXn2];

Xn2 = [Xn2_11;Xn2_12];

%Csample = unique([Xn1;Xn2],'rows');

Csample = unique([Csample; [Xn1;Xn2]],'rows');

disp(['length(Csample)=',num2str(length(Csample))]);

end

Xn1 = Csample(Csample(:,8)==1,:);
Xn2 = Csample(Csample(:,8)==-1,:);


%% COUNTERFACTUALS GENERATION: generation of the counterfactual example

X_NC = Xn1;

XC_NC = []; % NonCollision counterfactual
% Xn2=Csample(Csample(:,8)==-1,:);
Xn2=Xn2(:,1:7);
X_non_mod=[];

dist = [];

for i = 1 : size(X_NC,1)
    
    for j=1: size(Xn2,1)
        if ((Xn2(j,1)-X_NC(i,1)==0) &&...
            abs(Xn2(j,2)-X_NC(i,2))<=250/MAX2 && ...
            abs(Xn2(j,3)-X_NC(i,3))<=100/MAX3 && ...
            abs(Xn2(j,4)-X_NC(i,4))<=5/MAX4 && ...
            abs(Xn2(j,7)-X_NC(i,7))<=0.05/MAX7) 
            X_non_mod=[X_non_mod;Xn2(j,1:m)];
            %points sampled on the boundary with F0, m, d_ms, prob fixed
        end
    end
    
    if isempty(X_non_mod)== 1 % if there are no points satisfactoring the constraints
        A = [0,0,0,0,0,0,0];
        XC_NC = [XC_NC; A];
    else
        z = X_NC(i,1:m);
        K = KernelMatrix(X_non_mod,z,kernel, s);
        sq_dist = 2*(1-K);
        sq_minimum = min(sq_dist);
        [indexOpt,j] = find(sq_dist == sq_minimum);

        XC_NC = [XC_NC;X_non_mod(indexOpt(1),:)];

        zC = X_non_mod(indexOpt(1),:);

        dist_i = TestObject_N(Xtr, Ytr, x1, zC, kernel, s)-Rsquared1;

        dist = [dist;dist_i];

     end
    X_non_mod=[];

    disp(length(XC_NC));
    if length(XC_NC) > 100000
        break;
    end

end

counterlong = XC_NC;
pointslomg = X_NC;

XC_NC = counterlong;
X_NC = pointslomg;

ind_Cfcls = find(~all(XC_NC==0,2));
XC_NC = XC_NC(ind_Cfcls,:);
X_NC = X_NC(ind_Cfcls,:);

c = randperm(length(XC_NC));
c = c(1,1:size(XC_NC));

XC_NC = XC_NC(c, :);
X_NC = X_NC(c, :);
dist = dist(c,:);


yC_NC =TC_SVDD_TEST(Xtr, Ytr, alpha, XC_NC, kernel, s, Rsquared1, Rsquared2); %predicted class
XC_NC = [XC_NC, yC_NC];
XC_NC = [XC_NC(:,1)*MAX1, XC_NC(:,2)*MAX2, XC_NC(:,3)*MAX3,...
    XC_NC(:,4)*MAX4, XC_NC(:,5)*MAX5, XC_NC(:,6)*MAX6, XC_NC(:,7)*MAX7, XC_NC(:,8)];

X_NC = [X_NC(:,1:7), X_NC(:,8)];
X_NC = [X_NC(:,1)*MAX1, X_NC(:,2)*MAX2, X_NC(:,3)*MAX3,...
    X_NC(:,4)*MAX4, X_NC(:,5)*MAX5, X_NC(:,6)*MAX6, X_NC(:,7)*MAX7, X_NC(:,8)];

header1 = {'N','F0','m','d_ms','d0','v0','prob','class'};
body=[floor(X_NC(:,1)),floor(X_NC(:,2)),floor(X_NC(:,3)),floor(X_NC(:,4)),...
    floor(X_NC(:,5)),floor(X_NC(:,6)),round(X_NC(:,7),2), X_NC(:,8)];
writecell([header1; num2cell(body)],strcat(filepath,'collision_points.txt'),'Delimiter','tab');

header1 = {'N','F0','m','d_ms','d0','v0','prob','class'};
body=[floor(XC_NC(:,1)),floor(XC_NC(:,2)),floor(XC_NC(:,3)), ...
    floor(XC_NC(:,4)),floor(XC_NC(:,5)),floor(XC_NC(:,6)),round(XC_NC(:,7),2), XC_NC(:,8)];
writecell([header1; num2cell(body)],strcat(filepath,'collision_counterfactuals.txt'),'Delimiter','tab');

header1 = {'dist'};
body = [round(dist,2)];
writecell([header1; num2cell(body)],strcat(filepath,'count_distance.txt'),'Delimiter','tab');

%% Hystogram of the Counterfactual Quality

dist = sort(dist);

dist1 = dist(dist<0);
dist2 = dist(dist>=0 & dist<0.1);
dist3 = dist(dist>= 0.1);

bin_edges = -0.05:.01:0.18;

h1 = histogram(dist1,bin_edges,'FaceColor','r');

hold on 

h2 = histogram(dist2,bin_edges,'FaceColor','b');

hold on

h3 = histogram(dist3,bin_edges,'FaceColor','k');

xlabel('Counterfactual Quality','Interpreter','latex');
