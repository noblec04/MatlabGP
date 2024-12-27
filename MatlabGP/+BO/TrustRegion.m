function [x] = TrustRegion(Z)

lb = Z.lb_x;
ub = Z.ub_x;

dim = size(Z.X,2);

[YY,Ii] = sort(Z.Y,'descend');

nn = min(size(YY,1),50);

ii = randsample(nn,1);

xmax = Z.X(Ii(ii),:);

range = ub - lb;
lbn = xmax - range/5;
ubn = xmax + range/5;

xtest = lbn + (ubn - lbn).*lhsdesign(5,dim);

[mutest, vtest] = Z.eval(xtest);

vtest = sqrt(vtest);

[~,ii] = max(vtest.*(mutest+vtest));

x = xtest(ii,:);

end