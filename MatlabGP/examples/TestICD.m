
xx = -3 + 6*lhsdesign(1500,1);
yy = normrnd(sin(3*xx)./xx,0.1);

d = kernels.EQ(1,2);

[Kn,nn] = kernels.ICD(xx,d,9);

K = d.build(xx,xx(nn,:));

alpha = K\yy;

X2 = linspace(-3,3,100)';
K2 = d.build(X2,xx(nn,:));
Y2 = K2*alpha;

figure(1)
clf(1)
plot(X2,Y2)
hold on
plot(xx,yy,'.')
plot(xx(nn),yy(nn),'o')