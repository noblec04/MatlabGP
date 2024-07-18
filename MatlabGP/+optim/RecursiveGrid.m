function [x0,fmin,xv,fv] = RecursiveGrid(f,M,N,lb,ub,vectorize)

if nargin<6
    vectorize = 0;
end

dx = (ub-lb)/2;

x0 = 0.5*(lb+ub);

for i = 1:M

    lb1 = max(x0 - dx,lb);
    ub1 = min(x0 + dx,ub);

    x = lb1 + (ub1 - lb1).*lhsdesign(N,length(lb));

    if vectorize
        fn = f(x);
    else
        for j = 1:N
            fn(j) = f(x(j,:));
        end
    end

    [~,imin] = min(fn);

    x0 = x(imin,:);
    xv(i,:) = x0;

    dx = dx/5;

    fv(i) = fn(imin);

end

fmin = fn(imin);



end