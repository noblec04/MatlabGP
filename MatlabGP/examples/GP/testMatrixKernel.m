

clear all
close all
clc

xx = lhsdesign(100,2);

q = kernels.EQ_matrix(1,[2 0 3]);%.periodic(2,5);

q.signn = 0.001;

[K] = q.build(xx,xx);
%%
f1 = @(x) (6*x(:,1)-2).^2.*sin(12*x(:,2)-4);%.*sin(24*x-1);

x1 = lhsdesign(50,2);
y1 = f1(x1)+normrnd(0*x1(:,1),0*x1(:,1));

%%
a = means.const(0)+means.linear([2 2]);

q = kernels.EQ_matrix(1,[3 1 3]);%.periodic(2,5);
%q = kernels.EQ(1,[2 6]);%.periodic(2,5);
q.signn = eps;

Z = GP(a,q);

Z1 = Z.condition(x1,y1);

figure(1)
clf(1)
utils.plotSurf(Z1,2,1,'CI',false)

%%
tic
[Z2] = Z1.train();
toc
%%
figure
hold on
utils.plotSurf(Z2,2,1)
view(20,20)

%%

[x,R] = BO.argmax(@BO.maxGrad,Z2)

plot(x(1),x(2),'x','MarkerSize',18,'LineWidth',3)

plot3(x1(:,1),x1(:,2),y1,'+','MarkerSize',18)