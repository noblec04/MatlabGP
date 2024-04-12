

x0 = rand();

[x,Fx,xv,fv] = VSGD(@(x) forr(x),x0,'lr',0.1,'lb',0,'ub',1,'gamma',0.01,'iters',2000,'tol',1*10^(-5));

figure
plot(xv)

figure
plot(fv)

figure
plot(xv,fv)
hold on
plot(x,Fx,'X')

function [y,dy] = forr(x)

y = ((6*x-2).^2).*sin(12*x-4);
dy =12*(6*x-2).*(sin(12*x-4) + (6*x-2).*cos(12*x-4)) + normrnd(0*x,1);

end