
clear
clc

x = linspace(0,10,500);

y1 = tanh(0.1*x).*(sin(10*x) +0.5*cos(50*x + 3) + 0.1*sin(100*x+20) + 0.05*sin(200*x+20)).*exp(-0.1*x.^2);

y1 = normrnd(y1,0.05);

x2 = x-2;

y2 = tanh(0.1*x2).*(sin(10*x2) +0.3*cos(50*x2 + 3) + 0.2*sin(100*x2+20) + 0.10*sin(200*x2+20)).*exp(-0.1*x2.^2);

y2(x<2)=0;

y2 = 0.1*normrnd(y2,0.01);


figure
plot(x,y1)
hold on
plot(x,y2)

[xc,tau] = xcorr(abs(y1),abs(y2),'normalized');
figure
plot(tau,xc)

%%
m = 0;

XC = zeros(500,501,'like',y1);

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

%%

figure
pcolor([-250:250]/50,x,abs(XC))
shading flat
