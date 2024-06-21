function [alpha, dalpha] = FUNBOmin(Z,x)

ys = min(Z.Y);

gamma = 1;

%Calculate std at x
[varf,dvarf] = Z.eval_var(x);

[muf,dmuf] = Z.eval_mu(x);

sigf = sqrt(abs(varf));
dsigf = dvarf./(2*sigf+eps);

%Calculate beta value at x
beta = -1*(ys - muf + gamma*sigf)/sigf;
dbeta = -1*(-dmuf./sigf + (gamma - beta)*dsigf/sigf);

%Calculate expected improvement over current best measurement
alpha = -sigf*(beta*normcdf(beta)+normpdf(beta));
dalpha = -1*(-dsigf*alpha/sigf - sigf*dbeta*normcdf(beta));

end