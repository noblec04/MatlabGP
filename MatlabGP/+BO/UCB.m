function [alpha, dalpha] = UCB(Z,x)

[varf,dvarf] = Z.eval_var(x);

[muf,dmuf] = Z.eval_mu(x);

sigf = sqrt(abs(varf));
dsigf = dvarf./(2*sigf+eps);

alpha = -muf - 2*sigf;

dalpha = dmuf + 2*dsigf;

end