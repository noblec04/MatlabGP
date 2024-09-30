function [y] = softplus(x,beta)

if nargin==1
    beta=1;
end

if beta*x>10

    y=x;

else

    y = log(1+exp(beta*x))/beta;

end

end