function [alpha, dalpha] = maxVAR(Z,x)

%Calculate std at x
[varf,dvarf] = Z.eval_var(x);

alpha = -1*abs(varf);

dalpha = dvarf;

end