function [x,R] = argmax(FF,Z)

lb = Z.lb_x;
ub = Z.ub_x;

x0 = lb + (ub-lb).*rand(length(lb));

[x,R] = VSGD(@(x) FF(Z,x),x0,'lr',0.1,'lb',lb,'ub',ub,'gamma',0.01,'iters',2000,'tol',1*10^(-5));


end