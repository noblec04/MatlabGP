
clear
clc

A = AutoDiff([0.5 0.9 1.6 0.4 10 8]);

tic
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.PredatorPreyRHS(t,y,A(1),A(2),A(3),A(4)), [A(5);A(6)], 0, 100, 0.01,1e-6, A);

L = sum((yf{end} - [1;1]).^2);
toc


for i = 1:length(tf)
    yfv(:,i) = getvalue(yf{i});
    dyfv(:,:,i) = full(getderivs(yf{i}));
end

%%

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



