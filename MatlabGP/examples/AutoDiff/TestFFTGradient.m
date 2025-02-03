
clear
clc

x = linspace(0,10,1000);

omega = AutoDiff(2);

y = sin(omega*x).*exp(-x/omega);

figure
subplot(1,5,1)
plot(x,getvalue(y))

yh = fft(y);

E = yh.*conj(yh);
phi = imag(log(yh./conj(yh)));

subplot(1,5,2)
Ev = getvalue(E);
plot(Ev(1:end/2))
set(gca,'xscale','log')

dE = getderivs(E);

subplot(1,5,3)
plot(dE(1:end/2))
set(gca,'xscale','log')

subplot(1,5,4)
phiv = getvalue(phi);
plot(phiv(1:end/2))
set(gca,'xscale','log')

dphi = getderivs(phi);

subplot(1,5,5)
plot(dphi(1:end/2))
set(gca,'xscale','log')