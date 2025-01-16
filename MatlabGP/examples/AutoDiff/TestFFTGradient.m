
clear
clc

x = linspace(0,10,1000);

omega = AutoDiff(5);

y = sin(omega*x).*exp(-x/omega);

figure
subplot(1,3,1)
plot(x,getvalue(y))

yh = fft(y);

E = yh.*conj(yh);

subplot(1,3,2)
Ev = getvalue(E);
plot(Ev(1:end/2))
set(gca,'xscale','log')

dEmax = getderivs(E);

subplot(1,3,3)
plot(dEmax(1:end/2))
set(gca,'xscale','log')

