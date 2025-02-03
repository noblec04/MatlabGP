function [F] = Filter(F,iters)

% 8th order compact finite difference filter

if nargin<3
    iters=1;
end

alpha = .495;

a0 = (93 + 70*alpha)/128;
a1 = (7+18*alpha)/16;
a2 = (-7+14*alpha)/32;
a3 = (1-2*alpha)/16;
a4 = (-1+2*alpha)/128;

n = size(F,1);

for j = 1:iters

    f = F;

    for i = 5:n-4
        f(i,:) =  a0*F(i,:) + (a4/2)*(F(i+4,:)+F(i-4,:)) + (a3/2)*(F(i+3,:)+F(i-3,:)) + (a2/2)*(F(i+2,:)+F(i-2,:)) + (a1/2)*(F(i+1,:)+F(i-1,:));
    end

    f(1,:) = F(1,:);
    f(2,:) = .5*F(2,:) + .25*(F(1,:)+F(3,:));
    f(3,:) = .8125*F(3,:) + .125*(F(2,:)+F(4,:)) + -.03125*(F(1,:)+F(5,:));
    f(4,:) = .8125*F(4,:) + .125*(F(3,:)+F(5,:)) + -.03125*(F(2,:)+F(6,:));
    
    f(n-3,:) = .8125*F(n-3,:) + .125*(F(n-2,:)+F(n-4,:)) + -.03125*(F(n-1,:)+F(n-3,:));
    f(n-2,:) = .8125*F(n-2,:) + .125*(F(n-1,:)+F(n-3,:)) + -.03125*(F(n,:)+F(n-4,:));
    f(n-1,:) = .5*F(n-1,:) + .25*(F(n,:)+F(n-2,:)); 
    f(n,:) = F(n,:);

    a = 1*ones(n,1);
    b = alpha*ones(n,1);
    c = alpha*ones(n,1);

    b(1:4) = 0;
    c(1:4) = 0;

    b(n-3:n) = 0;
    c(n-3:n) = 0;

    if isa(f,'AutoDiff')
        F1.values = utils.tridiag( a, b, c, getvalue(f));
        F1.derivatives = f.derivatives;
        F = AutoDiff(F1);
    else
        F = utils.tridiag( a, b, c, f);
    end
end

end