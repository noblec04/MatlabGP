
clear
close all
clc

nD = 5;
nT = 1000;

xx = -2 + 4*[lhsdesign(nT,nD);utils.HypercubeVerts(nD)];

[yy] = testFuncs.Rosenbrock(xx,1)/7210;

xmesh = -2 + 4*lhsdesign(5000,nD);
ymesh = testFuncs.Rosenbrock(xmesh,1)/7210;

layers{1} = NN.FF(nD,nD);
layers{2} = NN.FF(nD,6);
layers{3} = NN.FF(6,1);

acts{1} = NN.SNAKE(8);
acts{2} = NN.SNAKE(8);

lss = NN.MSE();

nnet = NN.NN(layers,acts,lss);

%%

tic
[nnet2,fval] = nnet.Batchtrain(xx,yy,20);%,xv,fv
toc

%%

yp2 = nnet2.predict(xmesh);

%%

figure
plot(yp2,ymesh,'.')
