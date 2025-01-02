function [xn,YN,An] = MOUCB(Z,n0,q)

nD = length(Z.lb_x);

xm = Z.lb_x + (Z.ub_x - Z.lb_x).*lhsdesign(n0,nD);
YN = Z.UCB(xm);
An = utils.ParetoFront(YN);

xn = xm(An==1,:);

if size(xn,1)>=q

    iq = randsample(size(xn,1),q);
    xn = xn(iq,:);

end

end