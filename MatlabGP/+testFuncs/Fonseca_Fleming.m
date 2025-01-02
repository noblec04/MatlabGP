function [f] = Fonseca_Fleming(x)

d = size(x,2);

f(:,1) = 1 - exp(-1*sum((x - 1/sqrt(d)).^2));
f(:,2) = 1 - exp(-1*sum((x + 1/sqrt(d)).^2));


end