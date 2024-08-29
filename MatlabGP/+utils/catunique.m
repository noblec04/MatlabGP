function [x,flag] = catunique(x,xn)

reps = ismembertol(xn,x,1e-6,'ByRows',true);

if ~reps
    x = [x;xn];
end

flag = ~reps;

end