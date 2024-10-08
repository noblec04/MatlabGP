clear
close
clc

f1 = @(x) (6*x-2).^2.*sin(12*x-4);%.*sin(24*x-1);

xx = linspace(0,1,100)';
yy = f1(xx);

x1 = [0; 1*lhsdesign(5,1);1];
y1 = f1(x1)+normrnd(0*x1(:,1),0*x1(:,1)+0.05);

a = means.linear(4)*means.sine(1,3,0,1);

d = (kernels.Matern52(3,0.2).periodic(1,1) + kernels.EQ(0.2,0.4))*kernels.RQ(2,1,0.1);
%d = kernels.Matern52(1,0.1);
d.signn = 0;


Z = GP(a,d);

Z1 = Z.condition(x1,y1);

figure
utils.plotLineOut(Z1,1,1)
hold on
plot(xx,yy,'-.')
plot(x1,y1,'+','MarkerSize',12,'LineWidth',3)


figure
hold on

for i = 1:30
    ysamp = Z1.samplePrior(xx);
    plot(xx,ysamp,'LineWidth',0.05,'Color','k')
end

tic
[Z2] = Z1.train();
toc

%%
figure
hold on
utils.plotLineOut(Z2,1,1)
plot(xx,yy,'-.')
plot(x1,y1,'+','MarkerSize',12,'LineWidth',3)

