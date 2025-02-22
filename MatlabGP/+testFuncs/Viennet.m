function [f] = Viennet(x)

% -3<=x y<=3

y = x(:,2);
x = x(:,1);

f(:,1) = 0.5*(x.^2 + y.^2) + sin(x.^2 + y.^2);
f(:,2) = ((3*x - 2*y + 4).^2)/8 + ((x - y + 1).^2)/27 + 15;
f(:,3) = 1./(x.^2 + y.^2 + 1)  - 1.1*exp(-(x.^2 + y.^2));

end