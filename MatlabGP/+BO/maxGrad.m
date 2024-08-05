function [alpha, dalpha] = maxGrad(Z,x)

%Calculate std at x
[dmuf] = Z.eval_grad(x);

[dmuf2] = Z.eval_grad(x+0.05);

alpha = -1*norm(dmuf);

dalpha = 2*dmuf.*(dmuf2 - dmuf)/0.05;

end