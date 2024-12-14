function [d2F] = Lap(F,dx)

alpha = 2/11;

a = 12/11;
b = 3/44;

n = length(F);

f = F;
for i = 3:n-2
    f(i) =  a*(F(i+1)+F(i-1)) + b*(F(i+2)+F(i-2)) - (4*a + b)*F(i)/2;
end

f(1) = (13*F(1) - 27*F(2) + 15*F(3)  - F(4));
f(2) = (-3*F(1) - 10*F(2) + 18*F(3) - 6*F(4)+F(5))/12;

f(n-1) = (-1*F(n-4) + 6*F(n-3) - 18*F(n-2) + 10*F(n-1) + 3*F(n))/12;
f(n) = (3*F(n-4) - 16*F(n-3) + 36*F(n-2) - 48*F(n-1) + 25*F(n))/12;

a = 0*F + 1;
b = 0*F + alpha;
c = 0*F + alpha;

b(1) = 0;
b(2) = 0;
b(n-1) = 0;
b(n) = 0;

a(1) = 1;
a(2) = 1;
a(n-1) = 1;
a(n) = 1;


c(1) = 0;
c(2) = 0;
c(n-1) = 0;
c(n) = 0;


dF = utils.tridiag( a, b, c, f/dx^2 );

end