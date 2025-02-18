
clear
close all
clc

lb = [-2 -2];
ub = [2 2];

nmodes = 3;

%xx = utils.ndgrid(linspace(lb(1),ub(1),3),linspace(lb(2),ub(2),3));

xx = lb + (ub - lb).*[lhsdesign(18,2);utils.HypercubeVerts(2)];

xx = [xx;flipud(xx);xx;xx;xx;flipud(xx)];

for i = 1:size(xx,1)
    yy(i,:,:) = testFuncs.rosen_image_2D(xx(i,:),0.05);
end

ma = means.const(1);
ka = kernels.Matern52(1,[1 1]);
ka.signn = 1e-8;

kb = kernels.RQ(2,1,1);

Z = MOGP(ma,ka,nmodes);

pod1 = utils.KPOD(nmodes,kb);
%pod1 = utils.RPOD(nmodes);
%pod1 = utils.POD(nmodes);

PODgp = PODGP(Z,pod1);

%%

tic
PODgp = PODgp.train(xx,yy);
toc

%%

nn = 10;

[Xmesh,Ymesh] = ndgrid(linspace(-2,2,nn),linspace(-2,2,nn));

xmesh = [Xmesh(:) Ymesh(:)];

for i = 1:size(xmesh,1)
    mm1(:,:,i) = testFuncs.rosen_image_2D(xmesh(i,:),0);
    mm2(:,:,i) = PODgp.eval(xmesh(i,:));
    mm3(:,:,i) = PODgp.eval_var(xmesh(i,:));
    mm4(:,:,:,i) = PODgp.eval_grad(xmesh(i,:));
end

%%

figure
subplot(1,3,1)
imshow(imtile(mm1,'GridSize',[nn nn]))
utils.cmocean('thermal')
colorbar

%%

subplot(1,3,2)
imshow(imtile(mm2,'GridSize',[nn nn]))
utils.cmocean('thermal')
colorbar

%%

subplot(1,3,3)
imshow(imtile(mm3,'GridSize',[nn nn]))
utils.cmocean('thermal')
colorbar
clim auto

%%

figure
subplot(1,2,1)
imshow(imtile(squeeze(mm4(:,:,1,:)),'GridSize',[nn nn]))
utils.cmocean('thermal')
colorbar

subplot(1,2,2)
imshow(imtile(squeeze(mm4(:,:,2,:)),'GridSize',[nn nn]))
utils.cmocean('thermal')
colorbar


%%

utils.plotValidation(mm1(:),mm2(:))