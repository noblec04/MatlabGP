
clear
close all
clc

xx = [0;lhsdesign(1,1);1];
yy = forr(xx,0);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

%%

a = means.const(1);

b = kernels.EQ(1,2);
b.signn = eps;

%%
Z = GP(a,b);

%%
Z1 = Z.condition(xx,yy);

%%
tic
[Z2] = Z1.train();
toc

%%

utils.plotLineOut(Z2,0,1)

fprintf('Initial')
[E,V] = Z2.BayesQuad()
mean(ymesh)

for i = 1:length(xmesh)

    sc(i) = Z2.SquaredCorrelation(xmesh(i));

end

figure
plot(xmesh,sc)

for i = 1:8

    [xn,Rn] = BO.argmax(@BO.maxSC,Z2);
    yn = forr(xn,0);

    xx = [xx;xn];
    yy = [yy;yn];

    Z2 = Z2.condition(xx,yy);

    figure(3)
    clf(3)
    utils.plotLineOut(Z2,0,1)
    pause(0.1)
end

fprintf('Trained')
[E,V] = Z2.BayesQuad()
mean(ymesh)


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

