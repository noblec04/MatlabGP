clear
clc

x1 = lhsdesign(2000,2);
x2 = lhsdesign(1,2);

xs = lhsdesign(100,2);

a = kernels.RQ(2,1,1);

Kxx11 = a.build(x1,x1);
Kxx12 = a.build(x1,x2);
Kxx22 = a.build(x2,x2);

Kxs = a.build([x1;x2],xs);

Kxx = [Kxx11 Kxx12;
       Kxx12' Kxx22];

tic
Kxx11_inv = pinv(Kxx11);
toc

tic
Kxx_inv = pinv(Kxx);
toc

tic
Kxx_inv2 = utils.BlockPInv(Kxx11,Kxx12',Kxx22,Kxx12,Kxx11_inv);
toc