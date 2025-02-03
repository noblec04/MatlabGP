function [dF] = Deriv(F,dx)

% 6th order compact finite difference scheme (Lele et.al.)

alpha = 1;  % The Diagonal weight
beta = 1/3;    % The off-diagonal weight

a = 14/9 / 2;
b = 1/9 / 4;

n = size(F,1);

f = F;

I = 3:n-2;

f(I,:) = a*(F(I+1,:)-F(I-1,:)) + b*(F(I+2,:)-F(I-2,:));

f(1,:) = (-25*F(1,:) + 48*F(2,:) - 36*F(3,:) + 16*F(4,:) - 3*F(5,:))/12;
f(2,:) = (-3*F(1,:) - 10*F(2,:) + 18*F(3,:) - 6*F(4,:) + F(5,:))/12;

f(n-1,:) = (-1*F(n-4,:) + 6*F(n-3,:) - 18*F(n-2,:) + 10*F(n-1,:) + 3*F(n,:))/12;
f(n,:) = (3*F(n-4,:) - 16*F(n-3,:) + 36*F(n-2,:) - 48*F(n-1,:) + 25*F(n,:))/12;

a = alpha*ones(n,1);
b = beta*ones(n,1);
c = beta*ones(n,1);

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

dF = utils.tridiag( a, b, c, f/dx );

end