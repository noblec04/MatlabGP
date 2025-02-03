
clear
close all
clc

lb = [-2 -2];
ub = [2 2];

nmodes = 3;

%xx = utils.ndgrid(linspace(lb(1),ub(1),5),linspace(lb(2),ub(2),5));

xx = lb + (ub - lb).*[lhsdesign(10,2);utils.HypercubeVerts(2)];

xx = [xx;flipud(xx);xx;xx;xx;flipud(xx)];

for i = 1:size(xx,1)
    yy(i,:,:) = testFuncs.rosen_image_2D(xx(i,:),0.05);
end

layers{1} = NN.FAN(2,6,2);
layers{2} = NN.FF(6,6);
layers{3} = NN.FF(6,6);
layers{4} = NN.FF(6,nmodes);

acts{1} = NN.SWISH(1.2);
acts{2} = NN.SWISH(1.2);
acts{3} = NN.SWISH(0.8);

lss = NN.MAE();

nnet = NN.NN(layers,acts,lss);

pod1 = utils.POD(nmodes);

PODnn = PODNN(nnet,pod1);

%%

tic
PODnn = PODnn.train(xx,yy);
toc

%%

nn = 10;

[Xmesh,Ymesh] = ndgrid(linspace(-2,2,nn),linspace(-2,2,nn));

xmesh = [Xmesh(:) Ymesh(:)];

for i = 1:size(xmesh,1)
    mm1(:,:,i) = testFuncs.rosen_image_2D(xmesh(i,:),0);
    mm2(:,:,i) = PODnn.eval(xmesh(i,:));
end

%%

figure
subplot(2,1,1)
imshow(imtile(mm1,'GridSize',[nn nn]))
utils.cmocean('thermal')
colorbar

%%

subplot(2,1,2)
imshow(imtile(mm2,'GridSize',[nn nn]))
utils.cmocean('thermal')
colorbar


