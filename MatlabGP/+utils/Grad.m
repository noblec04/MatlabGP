function [dFy,dFx] = Grad(F,dy,dx)

if nargin == 1
    dy = 1;
end

dFy = utils.Deriv(F,dy);

if size(F,2)>1
    
    if nargin<3
        dx = dy;
    end

    dFx = utils.Deriv(F',dx)';

end