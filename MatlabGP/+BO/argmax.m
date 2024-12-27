function [x,R] = argmax(FF,Z,x0)

lb = Z.lb_x;
ub = Z.ub_x;

if nargin<3||isempty(x0)
    x0 = lb + (ub-lb).*rand(1,length(lb));
end

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