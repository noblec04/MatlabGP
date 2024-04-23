
nx = 50;

xx = lhsdesign(nx,5);

a = (kernels.Matern52(1,[3 4 3 2 1]) - kernels.EQ(1,[3 4 3 2 1]))*kernels.RQ(1,[3 4 3 2 1]) + kernels.EQ(1,[3 4 3 2 1]);

% ff = @(x) a.build(x,x2);
% 
% tic
% JAC = full(AutoDiffJacobianAutoDiff(ff,xx));
% toc

ff =@(x) buildK(a,xx,x);

tic
JAC = full(AutoDiffJacobianFiniteDiff(ff,a.getHPs()));
toc

%%


xx = lhsdesign(1,5);
x2 = lhsdesign(100,5);

tic
dKK = a.grad(xx,x2);
toc

%%

x = AutoDiff([2 3]');

f = (x+x)'*x;

%%

function K = buildK(ke,x,theta)

ke = ke.setHPs(theta);

K = ke.build(x,x);

end