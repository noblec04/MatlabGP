
clear
close all
clc

D = 2;

lb = -2*ones(1,D);
ub = 2*ones(1,D);

xx = lb + (ub - lb).*lhsdesign(500,D);
[yy,ee] = testFuncs.Rosenbrock_noisy(xx,1);
[yt] = testFuncs.Rosenbrock(xx,1);

%%

mb = means.zero();
b = kernels.Matern52(1,ones(1,D));
b.signn = 1e-2;

%%

tic
Z = HVGP(mb,b,[lhsdesign(100,D);utils.HypercubeVerts(D)]);
Z = Z.condition(xx,yy,ee,lb,ub);
Z = Z.train();
toc

%%

1 - mean((yt - Z.eval_mu(xx)).^2)./var(yt)
max(abs(yt - Z.eval_mu(xx)))./std(yt)


%%

tic
for i = 1:50
    Z = Z.addInducingPoints(Z.newXuDiff());
end
toc

%%

1 - mean((yt - Z.eval_mu(xx)).^2)./var(yt)
max(abs(yt - Z.eval_mu(xx)))./std(yt)