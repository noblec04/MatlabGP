
clear
close all
clc

xx = [0;lhsdesign(4,1);1];
yy = forr(xx,0);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

%%

a = means.const(1)+means.linear(1);

b = kernels.Matern52(0.5,5)+kernels.RQ(2,0.5,5);
b.signn = eps;

%%
Z = GP(a,b);

%%
Z1 = Z.condition(xx,yy);

%%
tic
V = Z1.getHPs();

opt = optim.AdamLS(V);
FF = @(x) Z1.loss(x); 

for i = 1:30
    
    Vi(:,i) = V;

    [e(i),dV] = Z1.loss(V);
    [opt,V] = opt.step(V,FF,dV);

end

Z2 = Z1.setHPs(V);
Z2 = Z2.condition(xx,yy);
toc

tic
Z3 = Z1.train();
toc

tic
Z4 = Z1.train2();
toc

%%

figure
utils.plotLineOut(Z2,1,1,'color','g')
hold on
plot(xx,yy,'.')

figure
utils.plotLineOut(Z3,1,1,'color','g')
hold on
plot(xx,yy,'.')

figure
utils.plotLineOut(Z4,1,1,'color','g')
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

