

clear all
close all
clc

f1 = @(x) (6*x-2).^2.*sin(12*x-4);%.*sin(24*x-1);

x1 = lhsdesign(2000,1);
y1 = f1(x1);

y1 = normrnd(y1,0.2*abs(y1)+0.01);

%%
a = means.const(0);

q = kernels.EQ(1,0.2);%+kernels.WN_Het(1,[1 1 1 1 1 1 1 1]);%.periodic(2,5);

q.signn = 0.1;

Z = VGP(a,q,linspace(0,1,8)');

Z1 = Z.condition(x1,y1);
%%
figure(1)
clf(1)
utils.plotLineOut(Z1,0.5,1,'CI',true)

%%
tic
[Z2] = Z1.train2(1);
toc
%%
figure(1)
clf(1)
utils.plotLineOut(Z2,0.5,1,'CI',true)
plot(x1,y1,'x')

%%
tic
[x,R] = BO.argmax(@BO.maxGrad,Z2)
toc

plot(x(1),0,'x','MarkerSize',18,'LineWidth',3)