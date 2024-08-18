
clear all
close all
clc

xx = [0;lhsdesign(8,1);1];
yy = forr(xx,0);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

%%

a = means.const(1)+means.linear(1);

b = kernels.EQ(1,2)+kernels.RQ(2,1,1)+kernels.Matern52(2,1)+kernels.RELU(1,1);
b.signn = eps;

%%
Z = GP(a,b);

%%
Z1 = Z.condition(xx,yy);

%%
tt = Z1.getHPs();

[ll,dll] = Z1.loss(tt);

%%
tic
[Z2] = Z1.train2();
toc

%%

tt2 = Z2.getHPs();

[ll2,dll2] = Z2.loss(tt2);

%%

utils.plotLineOut(Z2,1,1)
hold on
plot(xx,yy,'.')

%%

function y = forr(x,dx)

nx = length(x);

A = 0.5; B = 10; C = -5;

for i = 1:nx
    if x(i)<0.45
        y(i,1) = (6*x(i)-2).^2.*sin(12*x(i)-4);
    else
        y(i,1) = (6*x(i)-2).^2.*sin(12*x(i)-4)+dx;
    end
end

end

