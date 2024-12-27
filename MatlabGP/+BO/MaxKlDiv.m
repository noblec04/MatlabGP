function [alpha, dalpha] = MaxKlDiv(Z,Zi,x)

if nargout>1
    x=AutoDiff(x);
end

alpha = -1*utils.KL(Zi,Z,x);

if nargout>1
    dalpha = getderivs(alpha);
    alpha = getvalue(alpha);
end

end

