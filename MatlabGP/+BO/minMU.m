function [alpha, dalpha] = minMU(Z,x)

%Calculate std at x
[muf,dmuf] = Z.eval_mu(x);

alpha = muf;

dalpha = -1*dmuf;

end