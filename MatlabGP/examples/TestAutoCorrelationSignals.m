
clear
clc

x = linspace(0,10,500);

y1 = tanh(0.1*x).*(sin(10*x) +0.5*cos(50*x + 3) + 0.1*sin(100*x+20) + 0.05*sin(200*x+20)).*exp(-0.1*x.^2);

y1 = normrnd(y1,0.01);

x2 = x-2;

y2 = tanh(0.1*x2).*(sin(10*x2) +0.3*cos(50*x2 + 3) + 0.2*sin(100*x2+20) + 0.10*sin(200*x2+20)).*exp(-0.1*x2.^2);

y2(x<2)=0;

y2 = 0.4*normrnd(y2,0.02);

x3 = x-4;

y3 = tanh(0.1*x3).*(sin(10*x3) +0.8*cos(50*x3 + 3) + 0.05*sin(80*x3+40) + 0.20*sin(200*x3+20)).*exp(-0.1*x3.^2);

y3(x<4)=0;

y3 = 0.1*normrnd(y3,0.02);


figure
subplot(2,3,1)
plot(x,y1)
hold on
plot(x,y2)
plot(x,y3)

[xc,tau] = xcorr(y1,y2,'normalized');
[xc2,tau2] = xcorr(y1,y3,'normalized');
subplot(2,3,2)
plot(10*tau/500,xc)
hold on
plot(10*tau2/500,xc2)

%%
m = 0;

for i = -250:250
    m=m+1;
    for j = 1:500

        try
            XC(j,m) = y1(j)*y2(j-i);
        catch
            XC(j,m) = 0;
        end
    end
end

m = 0;

for i = -250:250
    m=m+1;
    for j = 1:500
        
        try
            XC2(j,m) = y1(j)*y3(j-i);
        catch
            XC2(j,m) = 0;
        end
    end
end

subplot(2,3,3)
surf([-250:250]/50,x,abs(XC))
shading flat

subplot(2,3,4)
surf([-250:250]/50,x,abs(XC2))
shading flat

XCpsd = (fft(XC').*conj(fft(XC')))';

subplot(2,3,5)
pcolor([1:250]/50,x,log(XCpsd(:,1:250)))
shading flat
set(gca,'xscale','log')
utils.cmocean('thermal')

XCpsd2 = (fft(XC2').*conj(fft(XC2')))';

subplot(2,3,6)
pcolor([1:250]/50,x,log(XCpsd2(:,1:250)))
shading flat
set(gca,'xscale','log')
utils.cmocean('thermal')