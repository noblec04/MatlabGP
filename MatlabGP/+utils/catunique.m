function [x,flag] = catunique(x,xn,tol)

if nargin<3
    tol = 1e-6;
end

reps = ismembertol(xn,x,tol,'ByRows',true);

flag = ~reps;

if flag
    x = [x;xn];
end



end