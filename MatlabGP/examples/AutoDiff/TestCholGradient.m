
clear
clc

xmesh = lhsdesign(20,2);

xs = lhsdesign(100,2);

b = kernels.EQ(1,[1 1]);

tt = 0.001:0.01:2;

for i = 1:200

theta = AutoDiff([tt(i) 1]);

b = b.setHPs(theta);

KXX = b.build(xmesh,xmesh) + (1e-6)*eye(size(xmesh,1));

KXs = b.build(xmesh,xs);

[L,flag] = chol(KXX);

v = L\KXs;

LL = sum(sum(v*v',1));

lv(i) = LL.values;
ld(:,i) = LL.derivatives;

end
