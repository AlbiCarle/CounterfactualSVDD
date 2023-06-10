%code for TC-SVDD training
function [x, Rsquared1, Rsquared2, a1, a2, x1, x2, SV1, SV2, YSV1, YSV2]=...
    TC_SVDD_TRAINING_NEW(Xtr, Ytr, kernel, param, C1, C2, C3, C4, qdprg_opts)

N1=size(Ytr(Ytr==1),1);
N2=size(Ytr(Ytr==-1),1);

N=N1+N2;

%%%%%%%%%%%%%%%%%%% XXXX %%%%%%%%%%%%%%%%%%
          % L=-(1/2x'Hx+f'x)

K=KernelMatrix(Xtr, Xtr, kernel, param);

%L1
Ytr1=Ytr;

H1=Ytr1*Ytr1'.*K;
H1=H1+H1';
f1=Ytr1.*diag(K);

%L2
Ytr2=-Ytr;

H2=Ytr2*Ytr2'.*K;
H2=H2+H2';
f2=Ytr2.*diag(K);

% L=L1+L2

H=[H1 zeros(N,N); zeros(N,N) H2];

f=[f1; f2];

lb = zeros(2*N,1);

ub1 = ones(N,1);
    ub1(Ytr1==-1,1)=C3;
    ub1(Ytr1==+1,1)=C1;
ub2 = ones(N,1);
    ub2(Ytr2==+1,1)=C2;
    ub2(Ytr2==-1,1)=C4;
ub=[ub1; ub2];
    
Aeq1 = ones(1,N);
    Aeq1(1,Ytr1==-1)=-1;
    Aeq1(1,Ytr1==+1)=+1;
    Aeq1=[Aeq1, zeros(1,N)];
Aeq2 = ones(1,N);
    Aeq2(1,Ytr2==-1)=-1;
    Aeq2(1,Ytr2==+1)=+1;
    Aeq2=[zeros(1,N), Aeq2];
Aeq=[Aeq1; Aeq2];
beq=[1;1];

if isequal(qdprg_opts,'on')
options = optimset('Display', 'on');
elseif isequal(qdprg_opts,'off')
options = optimset('Display', 'off');
end

x = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

%%%%%%%%%%%%%%%%%%% XXXX %%%%%%%%%%%%%%%%%%

x1=x(1:N,:); x2=x(N+1:2*N,:);

alf_i=x1(Ytr1==+1); alf1_l=x1(Ytr1==-1);

alf_l=x2(Ytr2==+1); alf2_i=x2(Ytr2==-1);

%%%%%%%%%% XXX %%%%%%%%%

X1=Xtr(Ytr1==+1,:); X2=Xtr(Ytr2==+1,:);

a1=alf_i'*X1-alf1_l'*X2;

a2=alf_l'*X2-alf2_i'*X1;

%%%%%%%%% XXX %%%%%%%%%

inc=1E-5;

idxSV1=find(all(abs(x1)>inc,2));
SV1=Xtr(idxSV1,:); YSV1=Ytr1(idxSV1,:);

idxSV2=find(all(abs(x2)>inc,2));
SV2=Xtr(idxSV2,:); YSV2=Ytr2(idxSV2,:);

%%%%%%%%% XXX %%%%%%%%%

xy1=[x1,Ytr1];
idxC11=find(all(abs(xy1(:,1))>C1-inc & xy1(:,2)==+1,2)); 
idxC12=find(all(abs(xy1(:,1))>C3-inc & xy1(:,2)==-1,2)); % SV outside the sphere
idxC1=[idxC11; idxC12];

idxEssSV1=setxor(idxSV1,idxC1);

EssSV1=Xtr(idxEssSV1,:);

if(size(EssSV1,1) >0)
rand=randperm(size(EssSV1,1),1); 
x_s1=EssSV1(rand,:);

Rsquared1=TestObject_N(Xtr, Ytr, x1, x_s1, kernel, param);
else
    Rsquared1=0;
end

xy2=[x2,Ytr2];
idxC21=find(all(abs(xy2(:,1))>C2-inc & xy2(:,2)==+1,2)); 
idxC22=find(all(abs(xy2(:,1))>C4-inc & xy2(:,2)==-1,2)); % SV outside the sphere
idxC2=[idxC21; idxC22];

idxEssSV2=setxor(idxSV2,idxC2);

EssSV2=Xtr(idxEssSV2,:);

if(size(EssSV2,1) >0)
rand=randperm(size(EssSV2,1),1); 
x_s2=EssSV2(rand,:);

Rsquared2=TestObject_N(Xtr, -Ytr, x2, x_s2, kernel, param);
else
    Rsquared2=0;
end

%%%%%%%%% XXX %%%%%%%%%


