
clear
clc

xmesh = lhsdesign(10,2);

xs = lhsdesign(1000,2);

b = kernels.EQ(1,[1 1]);

tt = 0.001:0.01:2;

for i = 1:200

theta = AutoDiff([tt(i) 1]);

b = b.setHPs(theta);

KXX = b.build(xmesh,xmesh);

KXs = b.build(xmesh,xs);

v = KXX\KXs;

LL = sum(sum(v*v',1));

lv(i) = LL.values;
ld(:,i) = LL.derivatives;

end
