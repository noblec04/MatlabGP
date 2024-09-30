
clear
close all
clc

D = 3;

lb = -2*ones(1,D);
ub = 2*ones(1,D);

xx = lb + (ub - lb).*lhsdesign(5000,D);
yy = testFuncs.Rosenbrock(xx,1);


%%
mb = means.linear(ones(1,D));
b = kernels.RQ(2,1,ones(1,D));
b.signn = 0.1;

%%
tic
Z = VGP(mb,b,[lhsdesign(10,D);utils.HypercubeVerts(D)]);
Z = Z.condition(xx,yy,lb,ub);
Z = Z.train();
toc

%%

1 - mean((yy - Z.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Z.eval_mu(xx)))./std(yy)


%%

opts = optimoptions('fmincon','Display','off');

tic
for i = 1:50
    %[xn,rn(i)] = fmincon(@(x) Z.newXuDiff(x),rand(1,D),[],[],[],[],0*lb,0*ub + 1,[],opts);
    Z = Z.addInducingPoints(Z.newXuDiff());
end
toc

%%

1 - mean((yy - Z.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Z.eval_mu(xx)))./std(yy)