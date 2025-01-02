function [f] = Ackley(x)

y = x(:,2);
x = x(:,1);

f = -20*exp(-0.2*sqrt(x.^2 + y.^2)) - exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + exp(1) + 20;

end