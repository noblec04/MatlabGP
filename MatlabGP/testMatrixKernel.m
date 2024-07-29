

clear all
close all
clc

xx = lhsdesign(100,2);

q = kernels.EQ_matrix(1,[2 0 3]);%.periodic(2,5);

q.signn = 0;

[K, dK] = q.build(xx,xx);
%%
f1 = @(x) (6*x(:,1)-2).^2.*sin(12*x(:,2)-4);%.*sin(24*x-1);

x1 = lhsdesign(20,2);
y1 = f1(x1)+normrnd(0*x1(:,1),0*x1(:,1));

%%
a = means.const(0);

q = kernels.EQ_matrix(1,[1 1 1]);%.periodic(2,5);

q.signn = 0;

Z = GP(a,q);

Z1 = Z.condition(x1,y1);

figure(1)
clf(1)
utils.plotSurf(Z1,2,1)

%%
tic
[Z2] = Z1.train2();
toc
%%
figure
hold on
utils.plotSurf(Z2,2,1)
view(20,20)

%%

[x,R] = BO.argmax(@BO.maxGrad,Z2)

plot(x(1),x(2),'x','MarkerSize',18,'LineWidth',3)