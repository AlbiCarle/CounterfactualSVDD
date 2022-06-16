%code for testing TC-SVDD
function y = ...
    TC_SVDD_TEST(Xtr, Ytr, x, Xts, kernel, param, Rsquared1, Rsquared2)

y=zeros(size(Xts,1),1);

N=size(Xtr,1);

x1=x(1:N,:); x2=x(N+1:2*N,:);

k1=TestObject_N(Xtr, Ytr, x1, Xts, kernel, param);
k2=TestObject_N(Xtr, -Ytr, x2, Xts, kernel, param);

for i=1:size(Xts)
    if(k1(i)-Rsquared1<=0 && k2(i)-Rsquared2>0)
        y(i)=+1;
    elseif(k2(i)-Rsquared2<=0 && k1(i)-Rsquared1>0)
        y(i)=-1;
    elseif(k1(i)-Rsquared1<=0 && k1(i)-k2(i)<0)
        y(i)=+1;
    elseif(k1(i)-Rsquared1<=0 && k1(i)-k2(i)>0)
        y(i)=-1;
    else
        y(i)=2;
    end
end

end



          