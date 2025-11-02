function [dF] = Grad2(F,dx,dim)

if nargin < 3
    dim = 1;
end

if isa(F,'AutoDiff')

    nz = size(getderivs(F),2);

    Fv = getvalue(F);

    dF.values = utils.Grad2(Fv,dx,dim);

    for i = 1:nz
        q = full(reshape(F.derivatives(:,i),size(Fv)));
        p = utils.Grad2(q,dx,dim);
        dF.derivatives(:,i) = p(:);
    end

    dF = AutoDiff(dF);
    return
end

nn = size(F);

a = 1:length(nn);

a(dim) = 1;

a(1) = dim;

F1 = permute(F,a);

nn2 = size(F1);

F2 = F1(:,:);

dF = utils.Deriv(F2,dx);

dF = reshape(dF,nn2);

dF = permute(dF,a);

end