

f1 = @(x) 20*x(:,2).*sin(12*x(:,1)-4);

x1 = lhsdesign(80,2);
y1 = f1(x1);

a = means.zero();
b = kernels.EQ_special(1,[5 5]);
b.signn = 0.01;

Z = GP(a,b);

Z1 = Z.condition(x1,y1);

figure
utils.plotSurf(Z1,1,2)

tic
[Z2] = Z1.train2();
toc

figure
utils.plotSurf(Z2,1,2)
hold on 
plot3(x1(:,1),x1(:,2),y1,'.','MarkerSize',18)
