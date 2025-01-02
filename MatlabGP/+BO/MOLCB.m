function [xn,YN,An] = MOLCB(Z,n0,q)

nD = length(Z.lb_x);

xm = Z.lb_x + (Z.ub_x - Z.lb_x).*lhsdesign(n0,nD);
YN = Z.LCB(xm);
An = utils.ParetoFront(YN,-1);

xn = xm(An==1,:);

s2n = Z.eval_var(xn);

[~,iV] = sort(s2n(:,1));

xn = xn(iV,:);

if size(xn,1)>=q

    %iq = randsample(size(xn,1),q);
    xn = xn(1:q,:);

end

end