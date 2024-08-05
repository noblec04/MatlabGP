function [val] = Jj(lj)

val = 2*lj*(sqrt(pi/2)*erf(1/(sqrt(2)*lj)) + lj*(exp(1/(2*lj^2))-1));

end