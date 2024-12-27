function [y] = Forrester(x,i)

A = 0.5; B = 10; C = -5;

switch i
    case 1
        for jj = 1:size(x,1)
            if x(jj)>0.45
                y(jj,1) = (6*x(jj)-2).^2.*sin(12*x(jj)-4) + 4;
            else
                y(jj,1) = (6*x(jj)-2).^2.*sin(12*x(jj)-4);
            end
        end
    case 2
        y = 0.4*(6*x-2).^2.*sin(12*x-4)-x-1;
    case 3
        y = A*(6*x-2).^2.*sin(12*x-4)+B*(x-0.5)-C;
end



end