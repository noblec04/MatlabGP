function [alpha, dalpha] = FUNBOmin(Z,x)

if nargout>1
    x=AutoDiff(x);
end

ys = min(Z.Y);

gamma = 1;

%Calculate std at x
[varf] = Z.eval_var(x);
[muf] = Z.eval_mu(x);

if nargout>1
    dmuf = getderivs(muf);
    muf = full(getvalue(muf));

    dvarf = getderivs(varf);
    varf = full(getvalue(varf));
end

sigf = sqrt(abs(varf));
dsigf = dvarf./(2*sigf+eps);

%Calculate beta value at x
beta = -1*(ys - muf + gamma*sigf)/sigf;

%Calculate expected improvement over current best measurement
alpha = -sigf*(beta*normcdf(beta)+normpdf(beta));

if nargout>1
    dbeta = -1*(-dmuf./sigf + (gamma - beta)*dsigf/sigf);
    dalpha = -1*(-dsigf*alpha/sigf - sigf*dbeta*normcdf(beta));
end

end