function [alpha, dalpha] = EImin(Z,x)

if nargout>1
    x=AutoDiff(x);
end

ys = min(Z.Y);

%Calculate std at x
[varf] = Z.eval_var(x);
[muf] = Z.eval_mu(x);

if nargout>1
    dmuf = getderivs(muf);
    muf = getvalue(muf);
end

sigf = sqrt(abs(varf));
%dsigf = dvarf./(2*sigf+eps);

if nargout>1
    dsigf = getderivs(sigf);
    sigf = getvalue(sigf);
end

%Calculate beta value at x
beta = -1*(ys - muf)/sigf;

%Calculate expected improvement over current best measurement
alpha = sigf*(beta*normcdf(beta)+normpdf(beta));

if nargout>1
    dalpha = dmuf.*normcdf(beta) + dsigf.*normpdf(beta);
end

end