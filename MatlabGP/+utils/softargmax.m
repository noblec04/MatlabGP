function [y] = softargmax(x,alpha)

if nargin==1
    alpha=1;
end

y = exp(alpha*x)./sum(exp(alpha*x),2);

end