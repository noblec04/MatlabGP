function I = KL(Z1,Z2,x)

[mu1] = Z1.eval(x);
[var1] = Z1.eval_var(x);

sig1 = sqrt(var1)';

[mu2] = Z2.eval(x);
[var2] = Z2.eval_var(x);

sig2 = sqrt(var2)';

I = abs(log(sig2./(sig1+eps)) + (sig1.^2 + (mu2-mu1).^2)./(2*sig2.^2+eps) - 1/2);

end