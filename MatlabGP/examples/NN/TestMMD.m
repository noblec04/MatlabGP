
clear
clc

x = randn(100,1);

y = normrnd(2,1,100,1);

x = AutoDiff(x);

loss = NN.MMD(1);

ee = loss.forward(x,y);

dee = full(getderivs(ee));