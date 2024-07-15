function [alpha, dalpha] = maxMU(Z,x)

%Calculate std at x
[muf,dmuf] = Z.eval_mu(x);

alpha = -1*muf;

dalpha = dmuf;

end