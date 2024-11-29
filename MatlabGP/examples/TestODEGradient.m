
clear
clc

A = AutoDiff([10,8/3,28]);

%{
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1 1 1], 0, 20, 0.01,  1e-4, A);

L = sum((yf(:,end) - [1;1;25]).^2);
%}

%{
[tf, yf] = ODE.rk38(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1 1 1], 0, 20, 0.01, A);

L = sum((yf(:,end) - [1;1;25]).^2);
%}

%{
[tf, yf] = ODE.heun(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1 1 1], 0, 20, 0.01, A);

L = sum((yf(:,end) - [1;1;25]).^2);
%}

[tf, yf] = ODE.dormandprince(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1 1 1], 0, 20, 0.01,1e-4,A);

L = sum((yf(:,end) - [1;1;25]).^2);

