function [dFy,dFx] = Grad(F,dy,dx)

if nargin == 1
    dy = 1;
end

if isa(F,'AutoDiff')

    nz = size(getderivs(F),2);

    Fv = getvalue(F);

    dFy.values = utils.Deriv(Fv,dy);

    for i = 1:nz
        q = reshape(F.derivatives(:,i),size(Fv));
        p = utils.Deriv(q,dy);
        dFy.derivatives(:,i) = p(:);
    end

    dFy = AutoDiff(dFy);
else

    dFy = utils.Deriv(F,dy);

end

if size(F,2)>1

    if nargin<3
        dx = dy;
    end

    if isa(F,'AutoDiff')

        nz = size(F.derivatives,2);

        Fv = getvalue(F);

        dFx.values = utils.Deriv(Fv',dx)';

        for i = 1:nz
            q = reshape(F.derivatives(:,i),size(Fv));
            p = utils.Deriv(q',dx)';
            dFx.derivatives(:,i) = p(:);
        end

        dFx = AutoDiff(dFx);
    else

        dFx = utils.Deriv(F,dx);

    end

end