
clear
clc

%%

x1 = lhsdesign(1000,6);

for i = 1:size(x1,1)
    [y1(i),dy1(i,:)] = testFunc(x1(i,:));
end

Dy = 0;
for i = 1:size(x1,1)

    Dy = Dy + dy1(i,:)'*dy1(i,:);

end

Dy = Dy/size(x1,1);

%%

[V,lam] = eig(Dy);

lam = lam/sum(diag(lam));

diag(lam)

W1 = V(:,[4:6]);

x2 = x1*W1;

x3 = x2*W1';


%%
function [y,dy] = testFunc(x)

if nargout==2
    x = AutoDiff(x);
end

y = (x(:,1).^2 + x(:,2).*x(:,3) + 0.01*x(:,3) - 0.1*cos(10*x(:,4)).*x(:,5))./(5 + exp(-2*(x(:,1)+x(:,6))));

if nargout==2
    dy = getderivs(y);
    y = getvalue(y);
end

end