
clear
close all
clc

xx = [lhsdesign(15,1)];
yy = normrnd(forr(xx,0),0*forr(xx,0));

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

ma = means.const(1);
ka = kernels.Matern52(1,1);

Z = MOGP(ma,ka,3);
Z = Z.condition(xx,yy,0,1);
Z = Z.train();

utils.plotLineOut(Z,0,1)
hold on
plot(xx,yy,'x')
plot(xmesh,ymesh,'.')

%%

[xn,Rn] = BO.argmax(@BO.HVUCB,Z)

%%

function y = forr(x,dx)

nx = length(x);

A = 0.5; B = 10; C = -5;

for i = 1:nx
    if x(i)<0.45
        y(i,1) = (6*x(i)-2).^2.*sin(12*x(i)-4);
    else
        y(i,1) = (6*x(i)-2).^2.*sin(12*x(i)-4)+dx;
    end

    y(i,2) = 0.4*(6*x(i)-2).^2.*sin(12*x(i)-4)-x(i)-1;
    y(i,3) = A*(6*x(i)-2).^2.*sin(12*x(i)-4)+B*(x(i)-0.5)-C;
end

end

