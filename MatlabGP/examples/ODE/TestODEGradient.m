
clear
clc

A = AutoDiff([10,8/3,28]);

tic
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1;1;1], 0, 10, 0.01,  1e-4, A);
toc

L = sum((yf{end} - [0;0;0]).^2);

for i = 1:length(tf)
    yfv(:,i) = getvalue(yf{i});
    dyfv(:,:,i) = full(getderivs(yf{i}));
end

figure
hold on
plot3(yfv(1,:),yfv(2,:),yfv(3,:))

%{
[tf, yf] = ODE.rk38(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1 1 1], 0, 20, 0.01, A);

L = sum((yf(:,end) - [1;1;25]).^2);
%}

%{
[tf, yf] = ODE.heun(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1 1 1], 0, 20, 0.01, A);

L = sum((yf(:,end) - [1;1;25]).^2);
%}

%{
[tf, yf] = ODE.dormandprince(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [1 1 1], 0, 20, 0.01,1e-4,A);

L = sum((yf(:,end) - [1;1;25]).^2);
%}

%%

tic
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.LorenzRHS(t,y,10,8/3,28), [1;1;1], 0, 100, 0.01,  1e-4);
toc

L = sum((yf{end} - [0;0;0]).^2);

for i = 1:length(tf)
    yfv(:,i) = yf{i};
end

plot3(yfv(1,:),yfv(2,:),yfv(3,:))