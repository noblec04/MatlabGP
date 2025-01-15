
clear
clc

x = lhsdesign(10,2);

x = AutoDiff(x);

ka = kernels.Matern52(1,[1 2]);

K = ka.build(x,x);

dK = full(getderivs(K));
dK = reshape(dK,[10,10,20]);

[U,S,V] = svd(K);

dU = full(getderivs(U));
dU = reshape(dU,[10,10,20]);

dS = full(getderivs(S));
dS = reshape(dS,[10,10,20]);