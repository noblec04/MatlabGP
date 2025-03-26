
clear
clc


tic
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.BushfireRHS(t,y), [0.3;0.2;0.5], 0, 100, 0.01,1e-6);

%L = sum((yf{end} - [1;1]).^2);
toc


for i = 1:length(tf)
    yfv(:,i) = yf{i};
    %yfv(:,i) = getvalue(yf{i});
    %dyfv(:,:,i) = full(getderivs(yf{i}));
end

%%

plot(tf,yfv)



