function [val] = Ijp(lj,xp)

val = sqrt(pi/2)*lj.*(erf(1./(sqrt(2).*lj).*(1-xp))+erf(1./(sqrt(2)*lj).*(xp)));

end