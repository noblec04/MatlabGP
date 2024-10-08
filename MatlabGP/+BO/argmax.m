function [x,R] = argmax(FF,Z)

lb = Z.lb_x;
ub = Z.ub_x;

x0 = lb + (ub-lb).*rand(1,length(lb));
try
    opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','off');

    [x,R] = fmincon(@(x) FF(Z,x),x0,[],[],[],[],lb,ub,[],opts);

catch

    opts = optimoptions('fmincon','SpecifyObjectiveGradient',false,'Display','off');

    [x,R] = fmincon(@(x) FF(Z,x),x0,[],[],[],[],lb,ub,[],opts);

end
%[x,R] = VSGD(@(x) FF(Z,x),x0,'lr',0.03,'lb',lb,'ub',ub,'gamma',0.01,'iters',100,'tol',1*10^(-3));

R = -1*R;

end