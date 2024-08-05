function [T1] = Term1(sigmap2,thetas,alpha,Kinv,X,s)

a=sigmap2;
for i = 1:size(X,2)
    if i~=s
        a = a*utils.sobol.Jj(thetas(i));
    end    
end

b=0;
for p = 1:size(X,1)
    for q = 1:size(X,1)
        b1 = (Kinv(p,q) - alpha(p)*alpha(q))*utils.sobol.Rpqs(X(p,s),X(q,s),thetas(s));

        for i = 1:size(X,2)
            if i~=s
                b1 = b1*utils.sobol.Ijp(thetas(i),X(p,i))*utils.sobol.Ijp(thetas(i),X(q,i));
            end
        end
        b = b+b1;
    end
end
b = b*sigmap2^2;

T1 = a - b;

end