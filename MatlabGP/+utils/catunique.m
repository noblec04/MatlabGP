function [x,flag] = catunique(x,xn)

reps = ismembertol(xn,x,1e-6,'ByRows',true);

flag = ~reps;

if flag
    x = [x;xn];
end



end