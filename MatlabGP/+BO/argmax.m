function [x,R] = argmax(FF,Z)

lb = Z.lb_x;
ub = Z.ub_x;

x0 = lb + (ub-lb).*rand(length(lb));

[x,R] = VSGD(@(x) FF(Z,x),x0,'lr',0.03,'lb',lb,'ub',ub,'gamma',0.01,'iters',100,'tol',1*10^(-5));

R = -1*R;

end