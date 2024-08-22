function [x,R] = argmin(FF,Z)%,xv,fv

lb = Z.lb_x;
ub = Z.ub_x;

for i = 1
    x0 = lb + (ub-lb).*rand(1,length(lb));
    try
        opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','none');

        [x{i},R(i)] = fmincon(@(x) FF(Z,x),x0,[],[],[],[],lb,ub,[],opts);

    catch
        opts = optimoptions('fmincon','SpecifyObjectiveGradient',false,'Display','none');

        [x{i},R(i)] = fmincon(@(x) FF(Z,x),x0,[],[],[],[],lb,ub,[],opts);
    end
    %[x{i},R(i),xv{i},fv{i}] = VSGD(@(x) FF(Z,x),x0,'lr',0.05,'lb',lb,'ub',ub,'gamma',1*10^(-2),'iters',100,'tol',1*10^(-6));
end

[~,ii] = min(R);

x = x{ii};
R = R(ii);
%xv = xv{ii};
%fv = fv{ii};

end