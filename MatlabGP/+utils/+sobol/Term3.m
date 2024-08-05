function [val] = Term3(sigmap2,thetas,alphas,X)

b = 1;
for i = 1:size(X,2)
    b = b.*utils.sobol.Ijp(thetas(i),X(:,i));   
end

b1 = (sigmap2^2)*b;

val = b1'*alphas;

end