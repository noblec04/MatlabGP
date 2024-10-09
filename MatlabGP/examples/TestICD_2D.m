
xx = lhsdesign(3000,2);
yy = normrnd((6*xx(:,1)-2).^2.*sin(12*xx(:,2)-4),2);

d = kernels.Matern52(1,[3 1]);

tic
[Kn,nn] = kernels.ICD(xx,d,25);
toc

K = d.build(xx,xx(nn,:));

tic
alpha = K\yy;
toc

[X12,X22] = ndgrid(linspace(0,1,100),linspace(0,1,100));

X2 = [X12(:) X22(:)];

K2 = d.build(X2,xx(nn,:));
Y2 = K2*alpha;

figure(1)
clf(1)
plot3(X2(:,1),X2(:,2),Y2,'.','MarkerSize',12)
hold on
plot3(xx(:,1),xx(:,2),yy,'.')
plot3(xx(nn,1),xx(nn,2),yy(nn),'o','MarkerSize',12,'LineWidth',3)