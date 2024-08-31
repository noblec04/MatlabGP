function [x,R] = argmaxGrid(FF,Z)

lb = Z.lb_x;
ub = Z.ub_x;

D = length(lb);

XT = lb + (ub - lb).*[lhsdesign(1000*D,D);utils.HypercubeVerts(D)];

[~,im] = max(FF(Z,XT));

x0 = XT(im,:);

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