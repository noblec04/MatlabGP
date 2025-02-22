function [alpha, dalpha] = FUNBOmax(Z,x)

if nargout>1
    x=AutoDiff(x);
end

ys = max(Z.Y);

gamma = 1;

%Calculate std at x
[muf] = Z.eval_mu(x);
[varf] = Z.eval_var(x);

if nargout>1
    dmuf = getderivs(muf);
    muf = full(getvalue(muf));

    dvarf = getderivs(varf);
    varf = full(getvalue(varf));
end

sigf = sqrt(abs(varf));

if nargout>1
    dsigf = dvarf./(2*sigf+eps);
end

%Calculate beta value at x
beta = (ys - muf + gamma*sigf)/sigf;

%Calculate expected improvement over current best measurement
alpha = -sigf*(beta*normcdf(beta)+normpdf(beta));

if nargout>1
    dbeta = (-dmuf./sigf + (gamma - beta)*dsigf/sigf);
    dalpha = -1*(-dsigf*alpha/sigf - sigf*dbeta*normcdf(beta));
end

end