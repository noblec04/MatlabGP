
clear
clc

A = AutoDiff([1 1 0]);

tic
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.VanDerPolRHS(t,y,A(1)), [A(2);A(3)], 0, 10, 0.01,  1e-7, A);
toc

L = sum((yf{end} - [1;1]).^2);

%{
tic
[tf, yf] = ODE.feuler(@(t,y) ODE.test.VanDerPolRHS(t,y,A(1)), [A(2);A(3)], 0, 10, 0.001, A);

L = sum((yf{end} - [1;1]).^2);
toc
%}

for i = 1:length(tf)
    yfv(:,i) = getvalue(yf{i});
    dyfv(:,:,i) = full(getderivs(yf{i}));
end

plot(tf,yfv)

%{
[tf, yf] = ODE.heun(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1 1 1], 0, 20, 0.01, A);

L = sum((yf(:,end) - [1;1;25]).^2);
%}

%{
[tf, yf] = ODE.dormandprince(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1 1 1], 0, 20, 0.01,1e-4,A);

L = sum((yf(:,end) - [1;1;25]).^2);
%}

