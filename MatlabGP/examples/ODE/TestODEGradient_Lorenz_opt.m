
clear
clc

opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','iter');

[xv,Rv] = fmincon(@(x) objective(x),[10 8/3 28 1 1 1],[],[],[],[],[0 0 0 -10 -10 -10],[30 30 30 10 10 10],[],opts);

%%

A = AutoDiff([xv(1) xv(2) xv(3) xv(4) xv(5) xv(6)]);

[tf, yf] = ODE.rkf45(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [A(4);A(5);A(6)], 0, 10, 0.01,  1e-9, A);

L = sum((yf{end} - [0;0;0]).^2);

dL = full(getderivs(L));
L = getvalue(L);

for i = 1:length(tf)
    yfv(:,i) = getvalue(yf{i});
    dyfv(:,:,i) = full(getderivs(yf{i}));
end

figure
plot3(yfv(1,:),yfv(2,:),yfv(3,:))

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

%%

function [L,dL] = objective(x)

A = AutoDiff([x(1) x(2) x(3) x(4) x(5) x(6)]);

[~, yf] = ODE.rkf45(@(t,y) ODE.test.LorenzRHS(t,y,A(1),A(2),A(3)), [A(4);A(5);A(6)], 0, 10, 0.01,  1e-4, A);

L = sum((yf{end} - [0;0;0]).^2);

dL = full(getderivs(L));
L = getvalue(L);

end


