function [alpha, dalpha] = maxMU(Z,x)

if nargout>1
    x=AutoDiff(x);
end

%Calculate std at x
[muf] = Z.eval_mu(x);

alpha = -1*muf;

if nargout>1
    dalpha = getderivs(alpha);
    alpha = getvalue(alpha);
end

end