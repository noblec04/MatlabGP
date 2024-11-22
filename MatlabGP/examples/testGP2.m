
clear
close all
clc

xx = [0;lhsdesign(20,1);1];
yy = forr(xx,0);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

%%

a = means.const(1)+means.linear(1);

b = kernels.EQ(0.25,2)+kernels.RQ(2,0.25,1)+kernels.Matern52(0.25,1)+kernels.RELU(0.25,1);
b.signn = eps;

%%
Z = GP(a,b);

%%
Z1 = Z.condition(xx,yy);

%%
tt = Z1.getHPs();
[ll,dll] = Z1.loss(tt)

%%
tic
[Z2] = Z1.train2();
toc

tt = Z2.getHPs();
[ll,dll] = Z2.loss(tt)

%%
tic
[Z3] = Z1.train();
toc

tt = Z3.getHPs();
[ll,dll] = Z3.loss(tt)

%%

utils.plotLineOut(Z2,1,1,'color','g')
hold on
utils.plotLineOut(Z3,1,1,'color','b')
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

