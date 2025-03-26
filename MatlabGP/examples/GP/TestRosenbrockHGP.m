
clear
close all
clc

% D = 2;
% 
% lb = -2*ones(1,D);
% ub = 2*ones(1,D);

D = 1;

lb=0;
ub=1;

xx = lb + (ub - lb).*[lhsdesign(80,D);utils.HypercubeVerts(D)];
[yy,ee] = testFuncs.Forrester_noisy(xx,1);
[yt] = testFuncs.Forrester(xx,1);

%%

mb = means.linear(ones(1,D))*means.linear(ones(1,D)) + means.linear(ones(1,D)) + means.const(ones(1,D));

kb = kernels.Matern52(1,5);
kb.signn = 5e-1;

kc = kernels.Matern52(1,5);
kc.signn = 1e-10;

%%

tic
Z = HGP(mb,kb,kc);
Z = Z.condition(xx,yy,ee,lb,ub);
Z = Z.train();
toc

%%

1 - mean((yt - Z.eval_mu(xx)).^2)./var(yt)
max(abs(yt - Z.eval_mu(xx)))./std(yt)
