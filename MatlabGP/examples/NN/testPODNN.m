
clear
close all
clc

lb = -2;
ub = 2;

xmesh = linspace(lb,ub,100)';

xx = lb + (ub - lb).*lhsdesign(5,1);

xx = [xx;flipud(xx);xx;xx;xx;flipud(xx)];

yy = testFuncs.rosen_image(xx,0.01);

layers{1} = NN.FAN(1,6,2);
layers{2} = NN.FF(6,3);
layers{3} = NN.FF(3,3);

acts{1} = NN.SWISH(1.2);
acts{2} = NN.SWISH(0.8);

lss = NN.MAE();

nnet = NN.NN(layers,acts,lss);


%%

ka = kernels.RQ(2,1,1);

%%

%pod1 = utils.POD(3);
%pod1 = utils.RPOD(3);
pod1 = utils.KPOD(3,ka);

PODnn = PODNN(nnet,pod1);

%%

tic
PODnn = PODnn.train(xx,yy);
toc

%%
yp2 = PODnn.eval(xmesh);

%%

figure
n=0;
for i = 1:10:100
    n=n+1;
    subplot(2,10,n)
    imagesc(squeeze(yp2(i,:,:))');
    axis square
end

n=0;
for i = 1:10:100
    n=n+1;
    subplot(2,10,n+10)
    imagesc(squeeze(testFuncs.rosen_image(xmesh(i),0))');
    axis square
end

