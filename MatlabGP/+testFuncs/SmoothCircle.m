function [y] = SmoothCircle(x,x0,R)


y = exp(-1*((x(:,1)-x0(1)).^2 + (x(:,2)-x0(2)).^2 - R).^2);


end