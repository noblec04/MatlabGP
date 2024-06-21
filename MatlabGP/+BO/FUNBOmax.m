function [alpha, dalpha] = FUNBOmax(Z,x)

ys = max(Z.Y);

gamma = 1;

%Calculate std at x
[varf,dvarf] = Z.eval_var(x);

[muf,dmuf] = Z.eval_mu(x);

sigf = sqrt(abs(varf));
dsigf = dvarf./(2*sigf+eps);

%Calculate beta value at x
beta = (ys - muf + gamma*sigf)/sigf;
dbeta = -dmuf./sigf + (gamma - beta)*dsigf/sigf;

%Calculate expected improvement over current best measurement
alpha = -sigf*(beta*normcdf(beta)+normpdf(beta));
dalpha = -dsigf*alpha/sigf - sigf*dbeta*normcdf(beta);

end