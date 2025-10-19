
clear
close all
clc

%%

D = 2;

lb = -2*ones(1,D);
ub = 2*ones(1,D);

tic

V0 = lb + (ub - lb).*lhsdesign(D,1)';

%%

%GaussNewton

V = V0;

opt1 = optim.GaussNewton();

for i = 1:2000
    
    Vi(:,i) = V;

    [e(i),dV] = testFuncs.Rosenbrock(V,1);
    [opt1,V] = opt1.step(V,e(i),dV);
    
end

time_diff = toc

V1 = Vi;
e1 = e;