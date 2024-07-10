function [alpha, dalpha] = EImin(Z,x)

ys = min(Z.Y);

%Calculate std at x
[varf,dvarf] = Z.eval_var(x);

[muf,dmuf] = Z.eval_mu(x);

sigf = sqrt(abs(varf));
dsigf = dvarf./(2*sigf+eps);

%Calculate beta value at x
beta = -1*(ys - muf)/sigf;

%Calculate expected improvement over current best measurement
alpha = sigf*(beta*normcdf(beta)+normpdf(beta));

dalpha = dmuf.*normcdf(beta) + dsigf.*normpdf(beta);

end