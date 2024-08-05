function [val] = Term2(sigmap2,thetas,Kinv,X)

a = sigmap2;
for i = 1:size(X,2)
    a = a*utils.sobol.Jj(thetas(i));   
end

b = 1;
for i = 1:size(X,2)
    b = b.*utils.sobol.Ijp(thetas(i),X(:,i));   
end

b1 = (sigmap2^2)*dot(b,Kinv*b);

val = a - b1;

end