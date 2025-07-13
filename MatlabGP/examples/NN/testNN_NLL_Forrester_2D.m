
clear
close all
clc

xx = [lhsdesign(30,2); utils.HypercubeVerts(2)];
[yy,ee] = testFuncs.Forrester_2D_noisy(xx,1);

layers{1} = NN.FF(2,3);
layers{2} = NN.FF(3,3);
layers{3} = NN.FF(3,6);
layers{4} = NN.FF(6,2);

acts{1} = NN.SWISH(1.1);
acts{2} = NN.SWISH(2);
acts{3} = NN.SWISH(3);

lss = NN.NLL('normal');

nnet = NN.NN(layers,acts,lss);

%%

tic
[nnet2,fval] = nnet.train(xx,[yy ee]');%,xv,fv
toc
