

x1 = RandomVariable.RV('dist',normrnd(-2,0.5,[1,2000]),'res',3000);
x2 = RandomVariable.RV('dist',normrnd(0,1,[1,2000]),'res',3000);

xm = -10:0.05:10;

figure
subplot(7,1,1)
x1.plot(xm)

subplot(7,1,2)
x2.plot(xm)

x3 = x1.*x1;
subplot(7,1,3)
x3.plot(xm)

x4 = (x1.*x2).^x3;
subplot(7,1,4)
x4.plot(xm)

x5 = x4.*x4;
subplot(7,1,5)
x5.plot(xm)

x6 = x5./x2;
subplot(7,1,6)
x6.plot(xm)

x7 = acos(x6);
subplot(7,1,7)
x7.plot(xm)
