function [y] = softmax(x,alpha)

if nargin==1
    alpha=1;
end

y = log(sum(exp(alpha*x),2))/alpha;

end