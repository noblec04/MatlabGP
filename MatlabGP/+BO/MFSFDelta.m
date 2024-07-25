function [alpha, dalpha] = MFSFDelta(Z,x)

if nargout>1
    x=AutoDiff(x);
end

[varf_mf] = Z.eval_var(x);
[muf_mf] = Z.eval_mu(x);

[varf_1] = Z.GPs{1}.eval_var(x);
[muf_1] = Z.GPs{1}.eval_mu(x);

alpha = -1*((muf_mf - muf_1).^2 + varf_mf + varf_1);

if nargout>1
    dalpha = getderivs(alpha);
    alpha = getvalue(alpha);
end

end