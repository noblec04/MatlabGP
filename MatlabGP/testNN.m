
clear all
close all
clc

f1 = @(x) (6*x-2).^2.*sin(12*x-4);

xx = lhsdesign(10,1);
yy = f1(xx);

xmesh = linspace(0,1,100)';
ymesh = f1(xmesh);

layers{1} = NN.FF(1,10);
layers{2} = NN.FF(10,10);
layers{3} = NN.FF(10,1);

acts{1} = NN.SNAKE(8);
acts{2} = NN.SWISH(1);

nnet = NN.NN(layers,acts);

%%
tic
[nnet2,fval,xv,fv] = nnet.train(xx,yy);
toc

%%

for j = 1:length(xmesh)
    yp1(j) = nnet.forward(xmesh(j,:));
    yp2(j) = nnet2.forward(xmesh(j,:));
end

%%
figure
plot(fv,'.')
set(gca,'yscale','log')
set(gca,'xscale','log')

figure
plot(xmesh,ymesh)
hold on
plot(xmesh,yp1)
plot(xmesh,yp2)
plot(xx,yy,'x')

