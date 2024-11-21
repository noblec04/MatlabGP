clear
clc

xmesh = linspace(0,1,300);

f1 = @(x) 20*x.*sin(8*x-4);
f2 = @(x) ((5*x+1).^2).*sin(12*x-4);
f3 = @(x) ((-x+1).^2).*sin(3*x+4);

ymesh1 = f1(xmesh);
ymesh2 = f2(xmesh);
ymesh3 = f3(xmesh);

x1 = [0;lhsdesign(12,1);1];
y1 = f1(x1);

x2 = [0;lhsdesign(4,1);1];
y2 = f2(x2);

x3 = [0;lhsdesign(8,1);1];
y3 = f3(x3);

x1 = [x1 0*x1+1];
x2 = [x2 0*x2+2];
x3 = [x3 0*x3+3];

XX = [x1;x2;x3];
YY = [y1;y2;y3];

a = means.linear([1 1]);
b = kernels.Matern52(1,[5 5]);
b.signn = eps;

Z = GP(a,b);

Z1 = Z.condition(XX,YY);
Z1 = Z1.train();

xtest1 = linspace(0,1,100)';
xtest1 = [xtest1 0*xtest1+1];

xtest2 = linspace(0,1,100)';
xtest2 = [xtest2 0*xtest2+2];

xtest3 = linspace(0,1,100)';
xtest3 = [xtest3 0*xtest3+3];

ytest1 = Z1.eval(xtest1);
ytest2 = Z1.eval(xtest2);
ytest3 = Z1.eval(xtest3);

sigtest1 = Z1.eval_var(xtest1);
sigtest2 = Z1.eval_var(xtest2);
sigtest3 = Z1.eval_var(xtest3);

figure
subplot(1,3,1)
hold on
plot(xtest1(:,1),ytest1)
plot(xtest1(:,1),ytest1+2*sqrt(sigtest1),':')
plot(xtest1(:,1),ytest1-2*sqrt(sigtest1),':')
plot(x1(:,1),y1,'+')
plot(xmesh,ymesh1)

subplot(1,3,2)
hold on
plot(xtest2(:,1),ytest2)
plot(xtest2(:,1),ytest2+2*sqrt(sigtest2),'--')
plot(xtest2(:,1),ytest2-2*sqrt(sigtest2),'--')
plot(x2(:,1),y2,'+')
plot(xmesh,ymesh2)

subplot(1,3,3)
hold on
plot(xtest3(:,1),ytest3)
plot(xtest3(:,1),ytest3+2*sqrt(sigtest3),'--')
plot(xtest3(:,1),ytest3-2*sqrt(sigtest3),'--')
plot(x3(:,1),y3,'+')
plot(xmesh,ymesh3)
