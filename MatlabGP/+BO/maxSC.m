function [alpha, dalpha] = maxSC(Z,x)

if nargout>1
    x=AutoDiff(x);
end

%Calculate std at x
[varf] = Z.SquaredCorrelation(x);

alpha = -1*abs(varf);

if nargout>1
    dalpha = getderivs(alpha);
    alpha = getvalue(alpha);
end

end