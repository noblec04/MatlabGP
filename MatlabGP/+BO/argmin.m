function [x,R,xv,fv] = argmin(FF,Z)

lb = Z.lb_x;
ub = Z.ub_x;

for i = 1:1
    x0 = lb + (ub-lb).*rand(1,length(lb));
    [x{i},R(i),xv{i},fv{i}] = VSGD(@(x) FF(Z,x),x0,'lr',0.02,'lb',lb,'ub',ub,'gamma',1*10^(-1),'iters',100,'tol',1*10^(-4));
end

[~,ii] = min(R);

x = x{ii};
R = R(ii);
xv = xv{ii};
fv = fv{ii};

end