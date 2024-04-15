clear all
%close all
clc

f1 = @(x) (6*x-2).^2.*sin(12*x-4);

xx = linspace(0,1,100)';
yy = f1(xx);

x1 = [0; 1*lhsdesign(1000,1);1];
y1 = f1(x1);
y1 = y1 + normrnd(0,0*y1 + 0.8);

%a = (Matern52(1,0.2).periodic(1,2) + EQ(1,0.4))*RQ(1,0.1);
a = Matern52(1,0.2);
%%

a.signn = 0.5;

Z = VGP([],a,lhsdesign(10,1));

%%

Z1 = Z.condition(x1,y1);

[ys,sig] = Z1.eval(xx);

figure(2)
clf(2)
plot(xx,ys)
hold on
% plot(xx,ys+2*sqrt(sig),'--')
% plot(xx,ys-2*sqrt(sig),'--')
plot(xx,yy,'-.')
plot(x1,y1,'+')

%%

tic
[Z2,LL] = Z1.train(1);
toc

%%
[ys,sig] = Z2.eval(xx);

figure(3)
clf(3)
plot(xx,ys)
hold on
plot(xx,ys+2*sqrt(sig),'--')
plot(xx,ys-2*sqrt(sig),'--')
plot(xx,yy,'-.')
plot(x1,y1,'+')

%%

figure(4)
plot(xx,ys)
hold on
plot(xx,ys+2*sqrt(sig),'--')
plot(xx,ys-2*sqrt(sig),'--')

for i = 1:30
    ysamp = Z1.samplePosterior(xx);
    plot(xx,ysamp,'LineWidth',0.05,'Color','k')
end