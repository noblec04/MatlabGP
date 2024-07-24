function [alpha, dalpha] = MFSFDelta(Z,x)

[varf_mf,dvarf_mf] = Z.eval_var(x);
[muf_mf,dmuf_mf] = Z.eval_mu(x);

[varf_1,dvarf_1] = Z.GPs{1}.eval_var(x);
[muf_1,dmuf_1] = Z.GPs{1}.eval_mu(x);

alpha = -1*((muf_mf - muf_1).^2 + varf_mf + varf_1);
dalpha = 2*(muf_mf - muf_1).*(dmuf_mf - dmuf_1) + dvarf_mf + dvarf_1;

end