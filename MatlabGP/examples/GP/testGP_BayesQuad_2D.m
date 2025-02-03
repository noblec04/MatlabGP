
clear
close all
clc

xx = [lhsdesign(20,2);utils.HypercubeVerts(2)];
yy = forr(xx);

xmesh = lhsdesign(5000,2);
ymesh = forr(xmesh);

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

utils.plotSurf(Z2,[0 0],1,2)

fprintf('Initial')
[E,V] = Z2.BayesQuad()
mean(ymesh)

for i = 1:40

    [xn,Rn] = BO.argmax(@BO.maxSC,Z2);
    yn = forr(xn);

    xx = [xx;xn];
    yy = [yy;yn];

    Z2 = Z2.condition(xx,yy);

    figure(3)
    clf(3)
    utils.plotSurf(Z2,[0 0],1,2)
    pause(0.1)
end

fprintf('Trained')
[E,V] = Z2.BayesQuad()
mean(ymesh)


function y = forr(x)


y = (6*x(:,1)-2).^2.*sin(12*x(:,2)-4);


end

