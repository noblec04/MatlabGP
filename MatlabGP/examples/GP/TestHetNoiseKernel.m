
a = kernels.EQ(1,0.25);

H = diag(log([1 2 1 0.5 0.1 0.6 0.9 2]));

X = lhsdesign(200,2);
x = lhsdesign(8,2);

kxX = a.build(x,X);

K = diag(diag(exp(kxX'*H*kxX)));

figure(1)
clf(1)
plot3(x(:,1),x(:,2),diag(exp((H))),'x','MarkerSize',10,'LineWidth',3)
hold on
plot3(X(:,1),X(:,2),diag(K),'x','MarkerSize',10,'LineWidth',3)