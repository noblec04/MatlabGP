
clear
clc

opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','iter');

[xv,Rv] = fmincon(@(x) objective(x),[1 1 1],[],[],[],[],[0 -10 -10],[30 10 10],[],opts);


A = AutoDiff([xv(1) xv(2) xv(3)]);

[tf, yf] = ODE.rkf45(@(t,y) ODE.test.VanDerPolRHS(t,y,A(1)), [A(2);A(3)], 0, 10, 0.01,  1e-9, A);

L = sum((yf{end} - [0.5;-0.5]).^2);

dL = full(getderivs(L));
L = getvalue(L);

for i = 1:length(tf)
    yfv(:,i) = getvalue(yf{i});
    dyfv(:,:,i) = full(getderivs(yf{i}));
end

figure
plot(tf,yfv)

[nx,ny,~] = size(dyfv);

figure
k=0;
for i = 1:nx
    for j = 1:ny
        k=k+1;
        subplot(nx,ny,k)
        plot(tf,squeeze(dyfv(i,j,:)))
    end
end


function [L,dL] = objective(x)

A = AutoDiff([x(1) x(2) x(3)]);

[~, yf] = ODE.rkf45(@(t,y) ODE.test.VanDerPolRHS(t,y,A(1)), [A(2);A(3)], 0, 10, 0.01,  1e-4, A);

L = sum((yf{end} - [0.5;-0.5]).^2);

dL = full(getderivs(L));
L = getvalue(L);

end


