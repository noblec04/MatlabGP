function [F] = Filter(F,alpha,iters)

if nargin<3
    iters=1;
end

a0 = (93 + 70*alpha)/128;
a1 = (7+18*alpha)/16;
a2 = (-7+14*alpha)/32;
a3 = (1-2*alpha)/16;
a4 = (-1+2*alpha)/128;

n = length(F);

for i = 1:iters

    f = F;

    for i = 5:n-4
        f(i) =  a0*F(i) + (a4/2)*(F(i+4)+F(i-4)) + (a3/2)*(F(i+3)+F(i-3)) + (a2/2)*(F(i+2)+F(i-2)) + (a1/2)*(F(i+1)+F(i-1));
    end

    a = 0*F + 1;
    b = 0*F + alpha;
    c = 0*F + alpha;

    b(1:4) = 0;
    c(1:4) = 0;

    b(n-3:n) = 0;
    c(n-3:n) = 0;

    F = utils.tridiag( a, b, c, f);
end

end