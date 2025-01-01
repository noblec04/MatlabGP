function [x] = TrustRegion2(Z)

lb = Z.lb_x;
ub = Z.ub_x;

dim = size(Z.X,2);

[~,Ii] = max(Z.LOO);

xmax = Z.X(Ii,:);

range = ub - lb;
lbn = xmax - range/10;
ubn = xmax + range/10;

xtest = lbn + (ubn - lbn).*lhsdesign(20,dim);

[mutest, vtest] = Z.eval(xtest);

vtest = sqrt(vtest);

[~,ii] = max(vtest.*(mutest+vtest));

x = xtest(ii,:);

end