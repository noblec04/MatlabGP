function [y,e] = Forrester_2D_noisy(x,i)

A = 0.5; B = 10; C = -5;

switch i
    case 1
       y = (6*x(:,1)-2).^2.*sin(12*x(:,2)-4);

       x(:,1) = 0.5*x(:,1);

       e = abs((6*x(:,2)-2).^2.*sin(12*x(:,1)-4))+0.1;

       y = normrnd(y,sqrt(e));

    case 2
        y = 0.4*(6*x(:,1)-2).^2.*sin(12*x(:,2)-4)-x(:,2)-1;

        x(:,1) = 0.5*x(:,1);

        e = abs(0.4*(6*x(:,2)-2).^2.*sin(12*x(:,1)-4)-x(:,1)-1) + 0.1;

       y = normrnd(y,sqrt(e));

    case 3
        y = A*(6*x(:,1)-2).^2.*sin(12*x(:,2)-4)+B*(x(:,2)-0.5)-C;

        x(:,1) = 0.5*x(:,1);

        e = 0.1*abs(A*(6*x(:,2)-2).^2.*sin(12*x(:,1)-4)+B*(x(:,1)-0.5)-C) + 0.1;

        y = normrnd(y,sqrt(e));
end

end