function [y,e] = Forrester_noisy(x,i)

A = 0.5; B = 10; C = -5;

switch i
    case 1
       y = (6*x-2).^2.*sin(12*x-4);
       e = 0.05*y.^2 + 0.1;

       y = normrnd(y,sqrt(e));

    case 2
        y = 0.4*(6*x-2).^2.*sin(12*x-4)-x-1;
        e = 0.05*y.^2 + 0.1;

        y = normrnd(y,sqrt(e));

    case 3
        y = A*(6*x-2).^2.*sin(12*x-4)+B*(x-0.5)-C;
        e = 0.05*y.^2 + 0.1;

        y = normrnd(y,sqrt(e));
end

end