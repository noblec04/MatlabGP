function [val] = Term4(sigmap2,Kinv,thetas,alphas,X)

a = sigmap2;

b=0;
for p = 1:size(X,1)
    for q = 1:size(X,1)
        b1 = Kinv(p,q);

        for i = 1:size(X,2)
            b1 = b1*utils.sobol.Rpqs(X(p,i),X(q,i),thetas(i));
        end
        b = b+b1;
    end
end

b = (sigmap2^2)*b;

c=0;
for p = 1:size(X,1)
    for q = 1:size(X,1)
        c1 = alphas(p)*alphas(q);

        for i = 1:size(X,2)
            c1 = c1*utils.sobol.Rpqs(X(p,i),X(q,i),thetas(i));
        end
        c = c+c1;
    end
end

c = (sigmap2^2)*c;

val = a - b + c;

end