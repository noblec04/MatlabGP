function [y] = Rastrigin(x)

d = size(x,2);
A = 10;

y = 10*d + sum(x.^2 - A*cos(2*pi*x),2);

end