function [alpha, dalpha] = UCBSig(Z,x)

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

alpha = -1*(muf + 2*sigf).*sigf;

if nargout>1
    dalpha = -1*(dmuf + 2*dsigf).*sigf + alpha.*dsigf;
end

end