clear all
%close all
clc

f1 = @(x) (6*x-2).^2.*sin(12*x-4);


xx = linspace(0,1,100)';
yy = f1(xx);

x1 = [0; 1*lhsdesign(5,1);1];
y1 = f1(x1);
y1 = y1 + normrnd(0,0*y1 + 0.1);

a = EQ(3,0.1);

b = Matern52(2,0.2);

c = RQ(2,0.1);

d = (c+a)*b;

e = (c+a.periodic(1,5))*b.periodic([1],[2]);

K = e.build(xx,xx);

figure(1)
clf(1)
pcolor(K)

%%

Z = GP([],b);

%%

Z1 = Z.condition(x1,y1);

[ys,sig] = Z1.eval(xx);

figure(2)
clf(2)
plot(xx,ys)
hold on
plot(xx,ys+2*sqrt(sig),'--')
plot(xx,ys-2*sqrt(sig),'--')
plot(xx,yy,'-.')
plot(x1,y1,'+')

%%

Z2 = Z1.train();

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
