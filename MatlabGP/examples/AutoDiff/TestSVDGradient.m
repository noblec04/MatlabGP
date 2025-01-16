
clear
clc

N = 50;
nD = 2;

x = lhsdesign(N,nD);

x = AutoDiff(x);

ka = kernels.Matern32(1,[1 2]);

K = ka.build(x,x);

dK = full(getderivs(K));
dK = reshape(dK,[N,N,N*nD]);

[U,S,V] = svd(K);

dU = full(getderivs(U));
dU = reshape(dU,[N,N,N*nD]);

dV = full(getderivs(V));
dV = reshape(dV,[N,N,N*nD]);

dS = full(getderivs(S));
dS = reshape(dS,[N,N,N*nD]);