
clear
close all
clc

%%

D = 5;

lb = -20*ones(1,D);
ub = 20*ones(1,D);

tic

V0 = lb + (ub - lb).*lhsdesign(D,1)';

func = @(x) customFunc(x);

%%

%GaussNewton

V = V0;

opt1 = optim.GaussNewton();

for i = 1:2000
    
    Vi(:,i) = V;

    [f, J] = AutoDifffuncAndJac(func, V);
    [opt1,V] = opt1.step(V,f,J);

    e(:,i) = f;
    
end

time_diff = toc

V1 = Vi;
e1 = e;

%%

function r = customFunc(x)

r(1) = x(1) + 5*x(2).^2 + 20*x(3) + x(4) - x(5);
r(2) = (6*x(1) - x(2)).^2 + 0.5*(x(2) - x(3)).^3 - 2 - x(4); 
r(3) = (20*x(1) - 3*x(2)).^2 + 0.5*(x(2) - x(3)).^2 - 5 - 3*x(4); 

end