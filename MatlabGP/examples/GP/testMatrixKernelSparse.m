

clear all
close all
clc

xx = lhsdesign(100,2);

q = kernels.EQ_matrix(1,[2 1 3]);%.periodic(2,5);

q.signn = 0.1;

[K, dK] = q.build(xx,xx);
%%
f1 = @(x) (6*x(:,1)-2).^2.*sin(12*x(:,2)-4);%.*sin(24*x-1);

x1 = lhsdesign(1000,2);
y1 = f1(x1)+normrnd(0*x1(:,1),0*x1(:,1));

%%
a = means.const(0);

q = kernels.EQ(1,[2 4]);%.periodic(2,5);

q.signn = 0.001;

Z = VGP(a,q,lhsdesign(30,2));

Z1 = Z.condition(x1,y1);

figure(1)
clf(1)
utils.plotSurf(Z1,2,1,'CI',false)
plot3(x1(:,1),x1(:,2),y1,'x','MarkerSize',18)

%%
tic
[Z2] = Z1.train(1);
toc
%%
figure
hold on
utils.plotSurf(Z2,2,1)
plot3(x1(:,1),x1(:,2),y1,'x','MarkerSize',18)
view(20,20)

%%
tic
[x,R] = BO.argmax(@BO.maxGrad,Z2)
toc

plot(x(1),x(2),'x','MarkerSize',18,'LineWidth',3)