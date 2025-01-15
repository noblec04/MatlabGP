
clear
close all
clc

%%

D = 2;

lb = -2*ones(1,D);
ub = 2*ones(1,D);

tic

V = lb + (ub - lb).*lhsdesign(D,1)';

opt = optim.diffGrad(V,'lr',0.5);

for i = 1:2000
    
    Vi(:,i) = V;

    [e(i),dV] = testFuncs.Rosenbrock(V,1);
    [opt,V] = opt.step(V,dV);
    
    % figure(1)
    % clf(1)
    % plot(e)
    % set(gca,'yscale','log')
    % set(gca,'xscale','log')
    
end

toc

%%
figure
subplot(2,1,1)
plot(e)
set(gca,'xscale','log')
set(gca,'yscale','log')

subplot(2,1,2)
plot(Vi')