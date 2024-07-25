function [alpha, dalpha] = LCB(Z,x)

if nargout>1
    x=AutoDiff(x);
end

[varf] = Z.eval_var(x);

[muf] = Z.eval_mu(x);

if nargout>1
    dmuf = getderivs(muf);
    muf = full(getvalue(muf));
end

sigf = sqrt(abs(varf));

if nargout>1
    dsigf = getderivs(sigf);
    sigf = full(getvalue(sigf));
end

alpha = muf - 2*sigf;

if nargout>1
    dalpha = dmuf - 2*dsigf;
end

end