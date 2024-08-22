
clear all
clc

D = 3;

lb = -2*ones(1,D);
ub = 2*ones(1,D);

xx = lb + (ub - lb).*lhsdesign(50000,D);
yy = testFuncs.Rosenbrock(xx,1);

x1 = lb + (ub - lb).*lhsdesign(50000,D);
y1 = testFuncs.Rosenbrock(x1,1);

%%
ma = means.const(0);
ka = kernels.EQ(1,0.01);
ka.signn = eps;

%%
Z = KISSGP(ma,ka,100);

%%
tic
Z1 = Z.condition(x1,y1);
toc

%%

1 - mean((yy - Z1.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Z1.eval_mu(xx)))./std(yy)

%%
tic
[Z2] = Z1.train();
toc

%%

1 - mean((yy - Z2.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Z2.eval_mu(xx)))./std(yy)