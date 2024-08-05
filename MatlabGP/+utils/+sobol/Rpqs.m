function [val] = Rpqs(xp,xq,ls)

val = sqrt(pi/4)*ls*exp(-(1/(4*ls^2))*(xp-xq)^2)*(erf((1/(2*ls))*(2 - xp-xq)) + erf((1/(2*ls))*(xp+xq)));

end